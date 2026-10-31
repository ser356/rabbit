import Foundation

/// Clasificador de actos congelado (`benchmarks/ytactos_modelo.json`): regresión logística multinomial sobre el vector
/// de V-JEPA 2, entrenada solo con YouTube. Mismo cálculo que `ytacts.probabilities`.
public struct ActClassifier: Decodable {
    public let classes: [String]
    let mean: [Double]
    let scale: [Double]
    let coef: [[Double]]
    let intercept: [Double]

    /// El modelo incluido en el paquete.
    public static func bundled() throws -> ActClassifier {
        guard let url = Bundle.module.url(forResource: "ytactos_modelo", withExtension: "json") else {
            throw RabbitSenseError.missingResource("ytactos_modelo.json")
        }
        return try load(url)
    }

    public static func load(_ url: URL) throws -> ActClassifier {
        try JSONDecoder().decode(ActClassifier.self, from: Data(contentsOf: url))
    }

    public var dimension: Int { mean.count }

    /// Probabilidad de cada clase, en el orden de `classes`.
    public func probabilities(_ x: [Float]) -> [Double] {
        precondition(x.count == mean.count, "El vector tiene \(x.count) valores; el modelo espera \(mean.count)")
        var z = [Double](repeating: 0, count: x.count)
        for i in 0..<x.count { z[i] = (Double(x[i]) - mean[i]) / scale[i] }
        var scores = (0..<classes.count).map { k -> Double in
            var s = intercept[k]
            let w = coef[k]
            for i in 0..<z.count { s += z[i] * w[i] }
            return s
        }
        let top = scores.max() ?? 0
        for k in scores.indices { scores[k] = exp(scores[k] - top) }
        let total = scores.reduce(0, +)
        return scores.map { $0 / total }
    }

    /// Clase más probable y su probabilidad.
    public func predict(_ x: [Float]) -> (act: String, confidence: Double, probabilities: [String: Double]) {
        let p = probabilities(x)
        let best = p.indices.max { p[$0] < p[$1] } ?? 0
        return (classes[best], p[best], Dictionary(uniqueKeysWithValues: zip(classes, p)))
    }
}

public enum RabbitSenseError: Error, CustomStringConvertible {
    case missingResource(String)
    case model(String)

    public var description: String {
        switch self {
        case .missingResource(let name): return "Falta el recurso \(name)"
        case .model(let message): return message
        }
    }
}
