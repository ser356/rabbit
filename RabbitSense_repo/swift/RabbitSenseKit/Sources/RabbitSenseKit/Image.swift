import Foundation

/// Imagen de 8 bits con 3 canales intercalados. El orden de los canales (RGB o BGR) lo decide quien la crea; recortes y
/// reescalados no dependen de él.
public struct RGBImage: Equatable {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 3, "Se esperaban \(width * height * 3) valores")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    public init(width: Int, height: Int) {
        self.init(width: width, height: height, pixels: [UInt8](repeating: 0, count: width * height * 3))
    }

    @inline(__always) public func value(_ x: Int, _ y: Int, _ c: Int) -> UInt8 {
        pixels[(y * width + x) * 3 + c]
    }

    /// Recorte con los límites de la imagen (como el troceado de numpy: lo que cae fuera se descarta).
    public func cropped(left: Int, top: Int, side: Int) -> RGBImage {
        let x0 = max(0, min(width, left)), y0 = max(0, min(height, top))
        let x1 = max(x0, min(width, left + side)), y1 = max(y0, min(height, top + side))
        var out = RGBImage(width: x1 - x0, height: y1 - y0)
        for y in y0..<y1 {
            let src = (y * width + x0) * 3, dst = ((y - y0) * out.width) * 3
            out.pixels.replaceSubrange(dst..<(dst + out.width * 3), with: pixels[src..<(src + out.width * 3)])
        }
        return out
    }

    /// Lo mismo que `channels[..., ::-1]`: RGB ↔ BGR.
    public func swappedChannels() -> RGBImage {
        var out = self
        for i in stride(from: 0, to: pixels.count, by: 3) {
            out.pixels[i] = pixels[i + 2]
            out.pixels[i + 2] = pixels[i]
        }
        return out
    }
}

/// Redondeo de Python (`round`) y de OpenCV (`cvRound`): al par más cercano en los empates.
@inline(__always) func roundEven(_ v: Double) -> Int { Int(v.rounded(.toNearestOrEven)) }

/// Reescalado como `cv2.resize(..., interpolation=cv2.INTER_AREA)`: promedio por áreas al reducir y la interpolación
/// de OpenCV para INTER_AREA al ampliar.
public func resizeArea(_ image: RGBImage, width: Int, height: Int) -> RGBImage {
    if width == image.width && height == image.height { return image }
    let sx = Double(image.width) / Double(width), sy = Double(image.height) / Double(height)
    if sx >= 1 && sy >= 1 {
        let xs = areaWeights(source: image.width, target: width, scale: sx)
        let ys = areaWeights(source: image.height, target: height, scale: sy)
        var out = RGBImage(width: width, height: height)
        var row = [Double](repeating: 0, count: image.width * 3)
        for dy in 0..<height {
            for i in row.indices { row[i] = 0 }
            for (sy0, wy) in ys[dy] {
                let base = sy0 * image.width * 3
                for i in 0..<(image.width * 3) { row[i] += wy * Double(image.pixels[base + i]) }
            }
            for dx in 0..<width {
                for c in 0..<3 {
                    var acc = 0.0
                    for (sx0, wx) in xs[dx] { acc += wx * row[sx0 * 3 + c] }
                    out.pixels[(dy * width + dx) * 3 + c] = UInt8(clamping: roundEven(acc / (sx * sy)))
                }
            }
        }
        return out
    }
    let xs = areaUpscale(source: image.width, target: width), ys = areaUpscale(source: image.height, target: height)
    var out = RGBImage(width: width, height: height)
    for dy in 0..<height {
        let (y0, y1, fy) = ys[dy]
        for dx in 0..<width {
            let (x0, x1, fx) = xs[dx]
            for c in 0..<3 {
                let top = (1 - fx) * Double(image.value(x0, y0, c)) + fx * Double(image.value(x1, y0, c))
                let bottom = (1 - fx) * Double(image.value(x0, y1, c)) + fx * Double(image.value(x1, y1, c))
                out.pixels[(dy * width + dx) * 3 + c] = UInt8(clamping: roundEven((1 - fy) * top + fy * bottom))
            }
        }
    }
    return out
}

/// Por píxel de destino, los píxeles de origen que cubre y la parte de cada uno (reducción por áreas).
func areaWeights(source: Int, target: Int, scale: Double) -> [[(Int, Double)]] {
    (0..<target).map { d in
        let a = Double(d) * scale, b = min(Double(d + 1) * scale, Double(source))
        var weights: [(Int, Double)] = []
        var s = Int(a.rounded(.down))
        while Double(s) < b - 1e-12 {
            let w = min(b, Double(s + 1)) - max(a, Double(s))
            if w > 1e-12 && s < source { weights.append((s, w)) }
            s += 1
        }
        return weights
    }
}

/// Coeficientes de OpenCV para INTER_AREA al ampliar: vecino más cercano con mezcla lineal en los bordes de cada píxel.
func areaUpscale(source: Int, target: Int) -> [(Int, Int, Double)] {
    let scale = Double(source) / Double(target), inverse = Double(target) / Double(source)
    return (0..<target).map { d in
        let s = Int((Double(d) * scale).rounded(.down))
        var f = Double(d + 1) - Double(s + 1) * inverse
        f = f <= 0 ? 0 : f - f.rounded(.down)
        let a = max(0, min(source - 1, s)), b = max(0, min(source - 1, s + 1))
        return (a, b, f)
    }
}

/// Caja cuadrada fija (izquierda, arriba, lado) en píxeles.
public struct SquareBox: Equatable, Codable {
    public let left: Int
    public let top: Int
    public let side: Int

    public init(left: Int, top: Int, side: Int) {
        self.left = left
        self.top = top
        self.side = side
    }
}

/// `acts.square_around`: cuadrado centrado en el rectángulo, del doble de su lado mayor y dentro de la imagen.
public func squareAround(x0: Double, y0: Double, x1: Double, y1: Double, width: Int, height: Int, pad: Double = 1)
    -> SquareBox {
    let side = min(Int(max(x1 - x0, y1 - y0) * (1 + 2 * pad / 2)), max(width, height))
    let cx = (x0 + x1) / 2, cy = (y0 + y1) / 2
    let left = Int(min(max(cx - Double(side) / 2, 0), Double(max(0, width - side))))
    let top = Int(min(max(cy - Double(side) / 2, 0), Double(max(0, height - side))))
    return SquareBox(left: left, top: top, side: side)
}

/// `acts.square`: recorte con la caja (o la imagen entera), reducido a `size` sin deformar y centrado con bandas negras.
public func squareCrop(_ image: RGBImage, box: SquareBox?, size: Int) -> RGBImage {
    let crop = box.map { image.cropped(left: $0.left, top: $0.top, side: $0.side) } ?? image
    let scale = Double(size) / Double(max(crop.width, crop.height))
    let w = max(1, roundEven(Double(crop.width) * scale)), h = max(1, roundEven(Double(crop.height) * scale))
    let resized = resizeArea(crop, width: w, height: h)
    var tile = RGBImage(width: size, height: size)
    let y = (size - h) / 2, x = (size - w) / 2
    for row in 0..<h {
        let src = row * w * 3, dst = ((row + y) * size + x) * 3
        tile.pixels.replaceSubrange(dst..<(dst + w * 3), with: resized.pixels[src..<(src + w * 3)])
    }
    return tile
}

/// `ytacts.probe_frames`: `n` índices repartidos (np.linspace + round de Python).
public func probeIndices(count: Int, n: Int) -> [Int] {
    guard count > 0 else { return [] }
    if n == 1 { return [0] }
    return (0..<n).map { j in roundEven(Double(j) * Double(count - 1) / Double(n - 1)) }
}
