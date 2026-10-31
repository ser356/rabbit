import CoreGraphics
import RabbitSenseKit
import Vision

/// Sigue a Cinna entre fotogramas con el seguidor de objetos de Apple (Vision), que va en tiempo real. OWLv2 solo se
/// usa para colocarlo o recolocarlo.
final class Tracker {
    private let handler = VNSequenceRequestHandler()
    private var request: VNTrackObjectRequest?
    static let minimumConfidence: Float = 0.3

    var isTracking: Bool { request != nil }

    /// Empieza a seguir la caja (píxeles, origen arriba a la izquierda) en una imagen de `width` × `height`.
    func start(box: Detection, width: Int, height: Int) {
        let w = Double(width), h = Double(height)
        let normalized = CGRect(x: box.x0 / w, y: 1 - box.y1 / h, width: (box.x1 - box.x0) / w, height: (box.y1 - box.y0) / h)
        let observation = VNDetectedObjectObservation(boundingBox: normalized)
        let request = VNTrackObjectRequest(detectedObjectObservation: observation)
        request.trackingLevel = .accurate
        self.request = request
    }

    func stop() { request = nil }

    /// Caja en este fotograma, o nil si se ha perdido (entonces deja de seguir hasta que se recoloque).
    func track(_ image: CGImage) -> Detection? {
        guard let request else { return nil }
        do {
            try handler.perform([request], on: image)
        } catch {
            stop()
            return nil
        }
        guard let observation = request.results?.first as? VNDetectedObjectObservation,
              observation.confidence >= Self.minimumConfidence else {
            stop()
            return nil
        }
        request.inputObservation = observation
        let w = Double(image.width), h = Double(image.height), b = observation.boundingBox
        return Detection(score: Double(observation.confidence), x0: b.minX * w, y0: (1 - b.maxY) * h,
                         x1: b.maxX * w, y1: (1 - b.minY) * h)
    }
}
