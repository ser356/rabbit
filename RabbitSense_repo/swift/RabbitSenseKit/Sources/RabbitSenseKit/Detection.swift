import Foundation

/// Detección del conejo: puntuación y caja (x0, y0, x1, y1) en píxeles de la imagen.
public struct Detection: Equatable, Codable {
    public let score: Double
    public let x0: Double
    public let y0: Double
    public let x1: Double
    public let y1: Double

    public init(score: Double, x0: Double, y0: Double, x1: Double, y1: Double) {
        self.score = score
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    }
}

public let detectorThreshold = 0.2
public let separateIoU = 0.3

/// `ytacts.iou`.
public func iou(_ a: Detection, _ b: Detection) -> Double {
    let w = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
    let h = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    let union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - w * h
    return union > 0 ? w * h / union : 0
}

/// `ytacts.detections`: salida de OWLv2 (centro, ancho y alto relativos al cuadrado con relleno) → detecciones en
/// píxeles, de mayor a menor, sin las que solapan con una mejor.
/// - Parameters:
///   - scores: probabilidad por parche (ya con la sigmoide).
///   - boxes: 4 valores por parche (cx, cy, w, h), seguidos.
public func detections(scores: [Double], boxes: [Double], width: Int, height: Int,
                       threshold: Double = detectorThreshold, separate: Double = separateIoU) -> [Detection] {
    let side = Double(max(width, height))
    let order = scores.indices.sorted { scores[$0] > scores[$1] }
    var found: [Detection] = []
    for k in order {
        if scores[k] < threshold { break }
        let cx = boxes[4 * k] * side, cy = boxes[4 * k + 1] * side
        let w = boxes[4 * k + 2] * side, h = boxes[4 * k + 3] * side
        let box = Detection(score: scores[k],
                            x0: min(max(cx - w / 2, 0), Double(width)), y0: min(max(cy - h / 2, 0), Double(height)),
                            x1: min(max(cx + w / 2, 0), Double(width)), y1: min(max(cy + h / 2, 0), Double(height)))
        if box.x1 - box.x0 < 2 || box.y1 - box.y0 < 2 { continue }
        if found.allSatisfy({ iou(box, $0) < separate }) { found.append(box) }
    }
    return found
}

/// `ytacts.window_detection`: resumen de las detecciones de los fotogramas muestreados de una ventana.
public struct WindowDetection: Equatable {
    public let found: Int
    public let probed: Int
    public let multiple: Bool
    public let scoreMin: Double?
    /// Rectángulo unión (x0, y0, x1, y1) de la mejor detección de cada fotograma.
    public let union: [Double]?

    public init(_ perFrame: [[Detection]]) {
        let best = perFrame.compactMap { $0.first }
        found = best.count
        probed = perFrame.count
        multiple = perFrame.contains { $0.count > 1 }
        scoreMin = best.map(\.score).min()
        union = best.isEmpty ? nil : [best.map(\.x0).min()!, best.map(\.y0).min()!,
                                      best.map(\.x1).max()!, best.map(\.y1).max()!]
    }
}
