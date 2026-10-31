import AVFoundation
import CoreML
import Foundation
import RabbitSenseKit
import UserNotifications

/// Estado de la app: modelos, cámara, diario de hoy y avisos.
@MainActor
final class Monitor: ObservableObject {
    @Published var status = "Cargando modelos…"
    @Published var running = false
    @Published var ready = false
    @Published var cameraName = "—"
    @Published var today: [WindowRecord] = []
    @Published var latest: WindowRecord?
    @Published var alerts: [RabbitAlert] = []

    let journal: Journal
    private let camera = Camera()
    private var pipeline: Pipeline?
    private var rules = AlertRules()
    private var history: [[WindowRecord]] = []
    private var historyDay = Date.distantPast
    private var lastCheck = Date.distantPast
    private var sent: [String: Date] = [:]
    /// Mientras observa, el Mac no se duerme solo ni manda la app a App Nap (la pantalla sí puede apagarse).
    private var activity: NSObjectProtocol?

    static var support: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RabbitSense")
    }

    /// Carpeta de los modelos convertidos: ajuste guardado, la que fijó el script de construcción o la del repositorio.
    static var weights: URL {
        if let saved = UserDefaults.standard.string(forKey: "weights") { return URL(fileURLWithPath: saved) }
        if let built = Bundle.main.object(forInfoDictionaryKey: "RabbitSenseWeights") as? String {
            return URL(fileURLWithPath: built)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Developer/Code/rabbit/RabbitSense_repo/weights/coreml")
    }

    private let notifications = ForegroundNotifications()

    init() {
        journal = Journal(folder: Self.support.appendingPathComponent("diario"))
        today = journal.records(on: Date())
        UNUserNotificationCenter.current().delegate = notifications
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
        Task { await load() }
    }

    /// Carga los modelos fuera del hilo principal; la primera vez los compila y los guarda compilados.
    func load() async {
        let weights = Self.weights
        let cache = Self.support.appendingPathComponent("modelos")
        do {
            let pipeline = try await Task.detached(priority: .userInitiated) { () throws -> Pipeline in
                let owl = try compiled(weights.appendingPathComponent("owlv2.mlpackage"), cache: cache)
                let vjepa = try compiled(weights.appendingPathComponent("vjepa2.mlpackage"), cache: cache)
                let detector = try OwlDetector(url: owl)
                let encoder = try VJEPAEncoder(url: vjepa, meta: weights.appendingPathComponent("modelos.json"))
                let classifierURL = Bundle.main.url(forResource: "ytactos_modelo", withExtension: "json")
                let classifier = try classifierURL.map(ActClassifier.load) ?? ActClassifier.bundled()
                let analyzer = WindowAnalyzer(detect: { try detector($0) }, embed: { try encoder($0) },
                                              classifier: classifier, clipFrames: encoder.frames, clipSize: encoder.size)
                return Pipeline(detector: detector, analyzer: analyzer)
            }.value
            pipeline.onRecord = { [weak self] record in Task { @MainActor in self?.add(record) } }
            self.pipeline = pipeline
            ready = true
            status = "Listo"
        } catch {
            status = "No se pudieron cargar los modelos de \(weights.path): \(error)"
        }
    }

    func start() {
        guard let pipeline, !running else { return }
        AVCaptureDevice.requestAccess(for: .video) { granted in
            Task { @MainActor in
                guard granted else {
                    self.status = "Sin permiso de cámara: actívalo en Ajustes del Sistema › Privacidad › Cámara"
                    return
                }
                guard let device = Camera.preferred() else {
                    self.status = "No hay cámara. Acerca el iPhone al Mac (Continuity Camera)."
                    return
                }
                do {
                    self.camera.onFrame = { rgb, image, time in pipeline.receive(rgb, image, time) }
                    try self.camera.start(device: device)
                    pipeline.start()
                    self.activity = ProcessInfo.processInfo.beginActivity(
                        options: [.userInitiated, .idleSystemSleepDisabled], reason: "Observando a Cinna")
                    self.cameraName = self.camera.deviceName
                    self.running = true
                    self.status = "Observando"
                } catch {
                    self.status = "No se pudo abrir la cámara: \(error)"
                }
            }
        }
    }

    func stop() {
        camera.stop()
        pipeline?.stop()
        if let activity { ProcessInfo.processInfo.endActivity(activity) }
        activity = nil
        running = false
        status = "En pausa"
    }

    var session: AVCaptureSession { camera.session }

    private func add(_ record: WindowRecord) {
        try? journal.append(record)
        if !Calendar.current.isDate(record.start, inSameDayAs: today.first?.start ?? record.start) { today = [] }
        today.append(record)
        if record.act != "skipped" { latest = record }
        let now = Date()
        guard now.timeIntervalSince(lastCheck) >= 60 else { return }
        lastCheck = now
        if !Calendar.current.isDate(historyDay, inSameDayAs: now) {
            history = journal.history(before: now, days: 14)
            historyDay = now
        }
        alerts = rules.evaluate(today: today, history: history, now: now)
        for alert in alerts where now.timeIntervalSince(sent[alert.identifier] ?? .distantPast) >= 3600 {
            sent[alert.identifier] = now
            let content = UNMutableNotificationContent()
            content.title = alert.title
            content.body = alert.body
            content.sound = .default
            UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: "\(alert.identifier)-\(now.timeIntervalSince1970)",
                                                                         content: content, trigger: nil))
        }
    }
}

/// Modelo compilado (.mlmodelc) en la caché; se recompila si el paquete es más nuevo.
func compiled(_ package: URL, cache: URL) throws -> URL {
    let target = cache.appendingPathComponent(package.deletingPathExtension().lastPathComponent + ".mlmodelc")
    let manager = FileManager.default
    let packageDate = (try? manager.attributesOfItem(atPath: package.path)[.modificationDate] as? Date) ?? .distantPast
    let cachedDate = (try? manager.attributesOfItem(atPath: target.path)[.modificationDate] as? Date) ?? .distantPast
    if manager.fileExists(atPath: target.path) && cachedDate >= packageDate { return target }
    let fresh = try MLModel.compileModel(at: package)
    try manager.createDirectory(at: cache, withIntermediateDirectories: true)
    try? manager.removeItem(at: target)
    try manager.moveItem(at: fresh, to: target)
    return target
}

/// Muestra los avisos también con la app delante (si no, macOS solo los manda al centro de notificaciones).
final class ForegroundNotifications: NSObject, UNUserNotificationCenterDelegate {
    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification,
                                withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler([.banner, .sound, .list])
    }
}
