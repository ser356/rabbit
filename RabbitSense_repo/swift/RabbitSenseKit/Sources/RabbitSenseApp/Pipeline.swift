import CoreGraphics
import Foundation
import RabbitSenseKit

/// Cadena en directo: cada fotograma pasa por el seguidor; cada 2 s se cierra una ventana y se analiza en segundo
/// plano (V-JEPA 2 + clasificador). OWLv2 recoloca el seguidor al empezar, cuando se pierde y cada 15 s.
/// Si el análisis va por detrás, se analiza siempre la ventana más reciente y las intermedias quedan «sin analizar».
/// Todo su estado se toca solo desde su cola interna.
final class Pipeline: @unchecked Sendable {
    static let window = 2.0
    static let redetect = 15.0
    static let noCameraAfter = 4.0
    static let minimumScore = 0.3

    private let queue = DispatchQueue(label: "rabbitsense.pipeline")
    private let detection = DispatchQueue(label: "rabbitsense.detection", qos: .utility)
    private let analysis = DispatchQueue(label: "rabbitsense.analysis", qos: .utility)
    private let detector: OwlDetector
    private let analyzer: WindowAnalyzer
    private let tracker = Tracker()

    private struct Window {
        var start: Date
        var frames: [RGBImage] = []
        var boxes: [Detection?] = []
        var samples: [MaskSample] = []
    }

    private var current: Window?
    private var pending: Window?
    private var analyzing = false
    private var detecting = false
    private var lastDetection = Date.distantPast
    private var lastFrame: Date?
    private var timer: DispatchSourceTimer?

    /// Resultado de cada ventana (en la cola interna; quien lo use debe pasar a la principal).
    var onRecord: ((WindowRecord) -> Void)?
    /// Caja actual del seguidor, normalizada (0–1, origen arriba a la izquierda), y proporción ancho/alto del fotograma.
    var onBox: ((CGRect?, CGFloat) -> Void)?

    init(detector: OwlDetector, analyzer: WindowAnalyzer) {
        self.detector = detector
        self.analyzer = analyzer
    }

    func start() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + Self.window, repeating: Self.window)
        timer.setEventHandler { [weak self] in self?.checkCamera() }
        timer.resume()
        self.timer = timer
    }

    func stop() {
        timer?.cancel()
        timer = nil
        queue.async {
            if let window = self.current { self.close(window, end: Date()) }
            self.current = nil
            self.tracker.stop()
        }
    }

    func receive(_ rgb: RGBImage, _ image: CGImage, _ time: Date) {
        queue.async { self.process(rgb, image, time) }
    }

    private func process(_ rgb: RGBImage, _ image: CGImage, _ time: Date) {
        lastFrame = time
        if !detecting && (!tracker.isTracking || time.timeIntervalSince(lastDetection) >= Self.redetect) {
            detecting = true
            lastDetection = time
            detection.async {
                let found = (try? self.detector(rgb)) ?? []
                self.queue.async {
                    self.detecting = false
                    if let best = found.first, found.count == 1, best.score >= Self.minimumScore {
                        self.tracker.start(box: best, width: rgb.width, height: rgb.height)
                    } else if found.isEmpty {
                        self.tracker.stop()
                    }
                }
            }
        }
        let box = tracker.track(image)
        let w = Double(rgb.width), h = Double(rgb.height)
        onBox?(box.map { CGRect(x: $0.x0 / w, y: $0.y0 / h, width: ($0.x1 - $0.x0) / w, height: ($0.y1 - $0.y0) / h) },
               CGFloat(w / h))
        if let window = current, time.timeIntervalSince(window.start) >= Self.window {
            close(window, end: time)
            current = nil
        }
        var window = current ?? Window(start: time)
        let offset = time.timeIntervalSince(window.start)
        window.frames.append(rgb)
        window.boxes.append(box)
        window.samples.append(MaskSample(time: offset, x: box.map { ($0.x0 + $0.x1) / 2 }, y: box.map { ($0.y0 + $0.y1) / 2 },
                                         area: box.map { ($0.x1 - $0.x0) * ($0.y1 - $0.y0) }))
        current = window
    }

    /// Sin fotogramas durante un rato: ventanas «sin cámara», que nunca cuentan como reposo.
    private func checkCamera() {
        let now = Date()
        guard current == nil || now.timeIntervalSince(lastFrame ?? .distantPast) >= Self.noCameraAfter else { return }
        if let window = current {
            close(window, end: lastFrame ?? now)
            current = nil
        }
        if now.timeIntervalSince(lastFrame ?? .distantPast) >= Self.noCameraAfter {
            onRecord?(WindowRecord(start: now.addingTimeInterval(-Self.window), end: now, act: "no_camera"))
        }
    }

    private func close(_ window: Window, end: Date) {
        guard window.frames.count >= 2 else { return }
        var window = window
        let limit = end.timeIntervalSince(window.start) + 1e-6
        window.samples = window.samples.filter { $0.time <= limit }
        if analyzing {
            if let skipped = pending {
                onRecord?(WindowRecord(start: skipped.start, end: skipped.start.addingTimeInterval(Self.window), act: "skipped"))
            }
            pending = window
        } else {
            run(window, end: end)
        }
    }

    private func run(_ window: Window, end: Date) {
        analyzing = true
        let duration = end.timeIntervalSince(window.start)
        analysis.async {
            let result = try? self.analyzer.analyze(frames: window.frames, boxes: window.boxes, start: 0, end: duration)
            let displacement = FixedCamera.netDisplacement(window.samples, start: 0, end: duration)
            let record = WindowRecord(start: window.start, end: end, act: result?.act ?? "skipped",
                                      confidence: result?.confidence,
                                      displacement: result?.act == "not_visible" ? nil : displacement)
            self.queue.async {
                self.analyzing = false
                self.onRecord?(record)
                if let next = self.pending {
                    self.pending = nil
                    self.run(next, end: next.start.addingTimeInterval(Self.window))
                }
            }
        }
    }
}
