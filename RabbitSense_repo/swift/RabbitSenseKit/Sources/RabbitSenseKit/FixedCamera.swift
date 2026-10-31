import Foundation

/// Muestra de la máscara de Cinna con la cámara fija: instante, centro y área (en píxeles). Sin máscara, `x` e `y` son nil.
public struct MaskSample: Equatable {
    public let time: Double
    public let x: Double?
    public let y: Double?
    public let area: Double?

    public init(time: Double, x: Double?, y: Double?, area: Double?) {
        self.time = time
        self.x = x
        self.y = y
        self.area = area
    }
}

/// Desplazamiento con cámara fija (`fixedcam.py`; sin validar todavía): sin movimiento de cámara que descontar, la
/// posición es el centro de la máscara y el tamaño corporal, √(área).
public enum FixedCamera {
    public static let tolerance = 0.25

    /// Posición en un instante: interpolación entre las muestras válidas que lo rodean (si distan ≤ 2 × tolerancia) o
    /// la muestra válida más cercana a ≤ tolerancia.
    public static func position(_ samples: [MaskSample], at instant: Double, tolerance: Double = tolerance)
        -> (x: Double, y: Double)? {
        let valid = samples.filter { $0.x != nil && $0.y != nil }
        let before = valid.last { $0.time <= instant + 1e-9 }
        let after = valid.first { $0.time >= instant - 1e-9 }
        if let i = before, let j = after, j.time - i.time <= 2 * tolerance {
            let w = j.time == i.time ? 0 : (instant - i.time) / (j.time - i.time)
            return ((1 - w) * i.x! + w * j.x!, (1 - w) * i.y! + w * j.y!)
        }
        let near = valid.filter { abs($0.time - instant) <= tolerance }
        guard let j = near.min(by: { abs($0.time - instant) < abs($1.time - instant) }) else { return nil }
        return (j.x!, j.y!)
    }

    /// Desplazamiento neto entre `start` y `end` en tamaños corporales, o nil si falta posición o tamaño.
    /// - Parameter samples: ordenadas por tiempo.
    public static func netDisplacement(_ samples: [MaskSample], start: Double, end: Double,
                                       tolerance: Double = tolerance) -> Double? {
        guard let a = position(samples, at: start, tolerance: tolerance),
              let b = position(samples, at: end, tolerance: tolerance) else { return nil }
        let sizes = samples.compactMap { s -> Double? in
            guard let area = s.area, area > 0, s.time >= start - 1e-9, s.time <= end + 1e-9 else { return nil }
            return area.squareRoot()
        }.sorted()
        guard !sizes.isEmpty else { return nil }
        let n = sizes.count
        let unit = n % 2 == 1 ? sizes[n / 2] : (sizes[n / 2 - 1] + sizes[n / 2]) / 2
        return ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot() / unit
    }

    /// Centro y área de una máscara binaria (valores > 0), o nil si está vacía.
    public static func centroid(mask: [UInt8], width: Int, height: Int) -> (x: Double, y: Double, area: Double)? {
        var sx = 0.0, sy = 0.0, n = 0.0
        for y in 0..<height {
            for x in 0..<width where mask[y * width + x] > 0 {
                sx += Double(x)
                sy += Double(y)
                n += 1
            }
        }
        return n > 0 ? (sx / n, sy / n, n) : nil
    }
}
