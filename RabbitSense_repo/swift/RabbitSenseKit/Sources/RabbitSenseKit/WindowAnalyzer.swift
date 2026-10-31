import Foundation

/// Resultado de una ventana de 2 s.
public struct WindowResult: Codable, Equatable {
    public let start: Double
    public let end: Double
    /// Fotogramas muestreados (de 4) en los que el detector vio al conejo.
    public let found: Int
    public let box: SquareBox?
    /// "locomotion", "eating", "lying", "other" o "not_visible".
    public let act: String
    public let confidence: Double?
    public let probabilities: [String: Double]

    /// Comer no es fiable (falló en las dos pruebas con Cinna).
    public var reliable: Bool { act != "eating" }
}

/// Acto de una ventana de 2 s, como `auto.act_windows`: OWLv2 en 4 fotogramas repartidos; sin conejo, «no visible»;
/// si no, caja fija del doble del rectángulo unión, 16 recortes a V-JEPA 2 y el clasificador congelado.
public struct WindowAnalyzer {
    public static let probes = 4
    public let detect: (RGBImage) throws -> [Detection]
    public let embed: ([RGBImage]) throws -> [Float]
    public let classifier: ActClassifier
    public let clipFrames: Int
    public let clipSize: Int

    public init(detect: @escaping (RGBImage) throws -> [Detection], embed: @escaping ([RGBImage]) throws -> [Float],
                classifier: ActClassifier, clipFrames: Int = 16, clipSize: Int = 256) {
        self.detect = detect
        self.embed = embed
        self.classifier = classifier
        self.clipFrames = clipFrames
        self.clipSize = clipSize
    }

    /// - Parameter frames: todos los fotogramas de la ventana, en orden, reducidos a lado largo 640 y en RGB.
    public func analyze(frames: [RGBImage], start: Double, end: Double) throws -> WindowResult {
        let probe = probeIndices(count: frames.count, n: Self.probes)
        return try classify(frames: frames, summary: WindowDetection(try probe.map { try detect(frames[$0]) }),
                            start: start, end: end)
    }

    /// Igual, pero con las cajas de un seguidor en lugar de buscar al conejo en cada fotograma muestreado (modo en
    /// directo). `boxes[i]` es la caja del fotograma i, o nil si el seguidor lo había perdido.
    public func analyze(frames: [RGBImage], boxes: [Detection?], start: Double, end: Double) throws -> WindowResult {
        precondition(boxes.count == frames.count)
        let probe = probeIndices(count: frames.count, n: Self.probes)
        return try classify(frames: frames, summary: WindowDetection(probe.map { i in boxes[i].map { [$0] } ?? [] }),
                            start: start, end: end)
    }

    func classify(frames: [RGBImage], summary: WindowDetection, start: Double, end: Double) throws -> WindowResult {
        guard let union = summary.union, frames.count >= 2 else {
            return WindowResult(start: start, end: end, found: summary.found, box: nil, act: "not_visible",
                                confidence: nil, probabilities: [:])
        }
        let box = squareAround(x0: union[0], y0: union[1], x1: union[2], y1: union[3],
                               width: frames[0].width, height: frames[0].height)
        let clip = probeIndices(count: frames.count, n: clipFrames).map { squareCrop(frames[$0], box: box, size: clipSize) }
        let prediction = classifier.predict(try embed(clip))
        return WindowResult(start: start, end: end, found: summary.found, box: box, act: prediction.act,
                            confidence: prediction.confidence, probabilities: prediction.probabilities)
    }
}
