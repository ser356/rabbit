import Foundation

/// Lo que se guarda de cada ventana de 2 s: nunca vídeo, solo el resultado.
public struct WindowRecord: Codable, Equatable {
    public let start: Date
    public let end: Date
    /// "locomotion", "eating", "lying", "other", "not_visible", "no_camera" (sin imagen) o "skipped" (sin analizar
    /// porque el análisis iba por detrás).
    public let act: String
    public let confidence: Double?
    /// Desplazamiento neto en tamaños corporales (cámara fija, provisional), si hubo medida.
    public let displacement: Double?

    public init(start: Date, end: Date, act: String, confidence: Double? = nil, displacement: Double? = nil) {
        self.start = start
        self.end = end
        self.act = act
        self.confidence = confidence
        self.displacement = displacement
    }

    public var seconds: Double { end.timeIntervalSince(start) }
    public var visible: Bool { !["not_visible", "no_camera", "skipped"].contains(act) }
}

/// Totales de un conjunto de ventanas.
public struct ActivitySummary: Equatable {
    public var secondsByAct: [String: Double] = [:]
    public var visibleSeconds = 0.0
    public var measuredSeconds = 0.0
    public var displacement = 0.0

    /// Desplazamiento por minuto medido, o nil si no hubo medida.
    public var displacementPerMinute: Double? { measuredSeconds > 0 ? displacement / (measuredSeconds / 60) : nil }

    /// Parte del tiempo visible en un acto.
    public func share(_ act: String) -> Double? { visibleSeconds > 0 ? (secondsByAct[act] ?? 0) / visibleSeconds : nil }

    public init(_ records: [WindowRecord]) {
        for r in records {
            secondsByAct[r.act, default: 0] += r.seconds
            if r.visible { visibleSeconds += r.seconds }
            if let d = r.displacement {
                displacement += d
                measuredSeconds += r.seconds
            }
        }
    }
}

/// Diario en disco: un archivo JSON Lines por día (AAAA-MM-DD.jsonl), una línea por ventana.
public final class Journal {
    public let folder: URL
    let calendar: Calendar
    let encoder: JSONEncoder
    let decoder: JSONDecoder

    public init(folder: URL, calendar: Calendar = .current) {
        self.folder = folder
        self.calendar = calendar
        encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }

    func file(for day: Date) -> URL {
        let c = calendar.dateComponents([.year, .month, .day], from: day)
        return folder.appendingPathComponent(String(format: "%04d-%02d-%02d.jsonl", c.year!, c.month!, c.day!))
    }

    public func append(_ record: WindowRecord) throws {
        let url = file(for: record.start)
        var line = try encoder.encode(record)
        line.append(0x0A)
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
        } else {
            try line.write(to: url)
        }
    }

    public func records(on day: Date) -> [WindowRecord] {
        guard let text = try? String(contentsOf: file(for: day), encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { try? decoder.decode(WindowRecord.self, from: Data($0.utf8)) }
    }

    /// Manda todos los días del diario a la Papelera (se pueden recuperar desde allí). La línea base vuelve a cero.
    public func reset() throws {
        let manager = FileManager.default
        for file in try manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
        where file.pathExtension == "jsonl" {
            try manager.trashItem(at: file, resultingItemURL: nil)
        }
    }

    /// Días anteriores a `day` (hasta `days`), del más reciente al más antiguo; solo los que tienen datos.
    public func history(before day: Date, days: Int) -> [[WindowRecord]] {
        (1...max(1, days)).compactMap { k in
            calendar.date(byAdding: .day, value: -k, to: day).map(records(on:))
        }.filter { !$0.isEmpty }
    }
}

/// Aviso para mostrar en el Mac.
public struct RabbitAlert: Equatable {
    public let identifier: String
    public let title: String
    public let body: String
}

/// Reglas de aviso. Sin línea base (menos de `minimumDays` días anteriores con datos) solo avisa de problemas de
/// imagen; con ella, compara las últimas horas con las mismas horas de los días anteriores.
public struct AlertRules {
    public var noCameraMinutes = 5.0
    public var notVisibleMinutes = 60.0
    public var minimumDays = 7
    public var compareHours = 3.0
    /// Aviso si el desplazamiento por minuto de las últimas horas baja de esta fracción de la mediana habitual.
    public var activityRatio = 0.5
    /// Tiempo visible mínimo en las últimas horas para comparar (no se avisa por falta de datos).
    public var minimumVisibleMinutes = 30.0

    public init() {}

    /// Si un día sirve de línea base: al menos una hora viendo a Cinna. Los días de pruebas o sin cámara no cuentan
    /// (la regla de actividad ya los descarta porque no tienen desplazamiento en las mismas horas).
    public static func countsForBaseline(_ records: [WindowRecord]) -> Bool {
        ActivitySummary(records).visibleSeconds >= 3600
    }

    /// Minutos seguidos, contando hacia atrás desde la última ventana, en que se cumple `condition`.
    func trailingMinutes(_ records: [WindowRecord], _ condition: (WindowRecord) -> Bool) -> Double {
        var seconds = 0.0
        for r in records.sorted(by: { $0.start > $1.start }) {
            guard condition(r) else { break }
            seconds += r.seconds
        }
        return seconds / 60
    }

    /// Registros con la misma hora del día que [now − horas, now], trasladados al día de cada registro.
    func sameHours(_ records: [WindowRecord], now: Date, calendar: Calendar) -> [WindowRecord] {
        let seconds = { (d: Date) -> Double in d.timeIntervalSince(calendar.startOfDay(for: d)) }
        let to = seconds(now), from = to - compareHours * 3600
        return records.filter { r in
            let t = seconds(r.start)
            return from >= 0 ? (t >= from && t < to) : (t >= from + 86400 || t < to)
        }
    }

    public func evaluate(today: [WindowRecord], history: [[WindowRecord]], now: Date,
                         calendar: Calendar = .current) -> [RabbitAlert] {
        var alerts: [RabbitAlert] = []
        let noCamera = trailingMinutes(today) { $0.act == "no_camera" }
        if noCamera >= noCameraMinutes {
            alerts.append(RabbitAlert(identifier: "sin-camara", title: "Sin imagen de la cámara",
                                body: "Lleva \(Int(noCamera)) minutos sin imagen. ¿Se ha movido o apagado el iPhone?"))
        }
        let hidden = trailingMinutes(today) { $0.act == "not_visible" }
        if hidden >= notVisibleMinutes {
            alerts.append(RabbitAlert(identifier: "no-visible", title: "Cinna no se ve",
                                body: "Lleva \(Int(hidden)) minutos sin aparecer en la imagen."))
        }
        guard history.count >= minimumDays else { return alerts }
        let recent = ActivitySummary(sameHours(today, now: now, calendar: calendar).filter { $0.start <= now })
        guard recent.visibleSeconds >= minimumVisibleMinutes * 60, let current = recent.displacementPerMinute else {
            return alerts
        }
        let usual = history.compactMap { ActivitySummary(sameHours($0, now: now, calendar: calendar)).displacementPerMinute }
            .sorted()
        guard usual.count >= minimumDays else { return alerts }
        let median = usual.count % 2 == 1 ? usual[usual.count / 2] : (usual[usual.count / 2 - 1] + usual[usual.count / 2]) / 2
        if median > 0 && current < activityRatio * median {
            alerts.append(RabbitAlert(identifier: "actividad-baja", title: "Cinna se mueve menos de lo habitual",
                                body: String(format: "En las últimas %.0f h se ha movido un %.0f %% menos que a estas horas otros días.",
                                             compareHours, 100 * (1 - current / median))))
        }
        return alerts
    }
}
