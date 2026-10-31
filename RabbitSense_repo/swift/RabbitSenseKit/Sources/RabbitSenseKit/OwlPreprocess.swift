import Foundation

let clipMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
let clipStd: [Float] = [0.26862954, 0.26130258, 0.27577711]

/// Preprocesado de `Owlv2ImageProcessor` (transformers 5, versión torchvision), paso a paso:
/// 1. valores entre 0 y 1;
/// 2. relleno con 0 abajo y a la derecha hasta hacer un cuadrado;
/// 3. si se reduce, desenfoque gaussiano previo (σ = (factor − 1) / 2, núcleo 2⌈3σ⌉ + 1, bordes reflejados);
/// 4. reescalado bilineal sin antialias (align_corners = false) a `size` × `size`;
/// 5. normalización con la media y la desviación de CLIP.
/// - Parameter image: imagen RGB.
/// - Returns: (3, size, size) en orden de canales, filas y columnas.
public func owlPixels(_ image: RGBImage, size: Int = 960) -> [Float] {
    let side = max(image.width, image.height)
    var planes = [[Float]](repeating: [Float](repeating: 0, count: side * side), count: 3)
    for y in 0..<image.height {
        for x in 0..<image.width {
            for c in 0..<3 { planes[c][y * side + x] = Float(image.value(x, y, c)) / 255 }
        }
    }
    let factor = Double(side) / Double(size)
    if factor > 1 {
        let sigma = (factor - 1) / 2
        let kernel = gaussianKernel(sigma: sigma)
        for c in 0..<3 { planes[c] = blur(planes[c], width: side, height: side, kernel: kernel) }
    }
    var out = [Float](repeating: 0, count: 3 * size * size)
    let ys = bilinearIndices(source: side, target: size), xs = bilinearIndices(source: side, target: size)
    for c in 0..<3 {
        let plane = planes[c]
        for dy in 0..<size {
            let (y0, y1, ly) = ys[dy]
            for dx in 0..<size {
                let (x0, x1, lx) = xs[dx]
                let top = (1 - lx) * plane[y0 * side + x0] + lx * plane[y0 * side + x1]
                let bottom = (1 - lx) * plane[y1 * side + x0] + lx * plane[y1 * side + x1]
                let v = (1 - ly) * top + ly * bottom
                out[(c * size + dy) * size + dx] = (v - clipMean[c]) / clipStd[c]
            }
        }
    }
    return out
}

/// Índices y peso para el reescalado bilineal de PyTorch con align_corners = false.
func bilinearIndices(source: Int, target: Int) -> [(Int, Int, Float)] {
    let scale = Double(source) / Double(target)
    return (0..<target).map { d in
        let s = max(0, (Double(d) + 0.5) * scale - 0.5)
        let i0 = min(Int(s), source - 1)
        let i1 = i0 < source - 1 ? i0 + 1 : i0
        return (i0, i1, Float(s - Double(i0)))
    }
}

/// Núcleo gaussiano 1D de torchvision: 2⌈3σ⌉ + 1 puntos, normalizado.
func gaussianKernel(sigma: Double) -> [Float] {
    let size = 2 * Int((3 * sigma).rounded(.up)) + 1
    let half = Double(size - 1) / 2
    let values = (0..<size).map { i -> Double in
        let x = -half + Double(i)
        return exp(-0.5 * (x / sigma) * (x / sigma))
    }
    let total = values.reduce(0, +)
    return values.map { Float($0 / total) }
}

/// Desenfoque separable con bordes reflejados (modo «reflect» de PyTorch: sin repetir el borde).
func blur(_ plane: [Float], width: Int, height: Int, kernel: [Float]) -> [Float] {
    let half = kernel.count / 2
    func reflect(_ i: Int, _ n: Int) -> Int {
        guard n > 1 else { return 0 }
        var j = i
        while j < 0 || j >= n { j = j < 0 ? -j : 2 * (n - 1) - j }
        return j
    }
    var horizontal = [Float](repeating: 0, count: plane.count)
    for y in 0..<height {
        for x in 0..<width {
            var acc: Float = 0
            for k in 0..<kernel.count { acc += kernel[k] * plane[y * width + reflect(x + k - half, width)] }
            horizontal[y * width + x] = acc
        }
    }
    var out = [Float](repeating: 0, count: plane.count)
    for y in 0..<height {
        for x in 0..<width {
            var acc: Float = 0
            for k in 0..<kernel.count { acc += kernel[k] * horizontal[reflect(y + k - half, height) * width + x] }
            out[y * width + x] = acc
        }
    }
    return out
}
