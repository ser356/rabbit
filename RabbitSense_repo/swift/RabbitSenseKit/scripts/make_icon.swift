// Dibuja el icono de la app (la liebre de SF Symbols, en blanco sobre negro) en todos los tamaños de un .iconset.
// Uso: swiftc scripts/make_icon.swift -o build/make_icon && build/make_icon build/AppIcon.iconset
import AppKit

let folder = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

func render(_ pixels: Int) -> Data {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                               bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = NSSize(width: pixels, height: pixels)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let s = CGFloat(pixels) / 1024
    // Cuadrado redondeado de la retícula de iconos de macOS (824 de 1024, con 100 de margen).
    let tile = NSBezierPath(roundedRect: NSRect(x: 100 * s, y: 100 * s, width: 824 * s, height: 824 * s),
                            xRadius: 185 * s, yRadius: 185 * s)
    NSGradient(starting: NSColor(white: 0.17, alpha: 1), ending: NSColor(white: 0.02, alpha: 1))!.draw(in: tile, angle: -90)
    let config = NSImage.SymbolConfiguration(pointSize: 400, weight: .medium)
        .applying(NSImage.SymbolConfiguration(paletteColors: [.white]))
    if let hare = NSImage(systemSymbolName: "hare", accessibilityDescription: nil)?.withSymbolConfiguration(config) {
        let scale = min(580 * s / hare.size.width, 580 * s / hare.size.height)
        let w = hare.size.width * scale, h = hare.size.height * scale
        hare.draw(in: NSRect(x: (CGFloat(pixels) - w) / 2, y: (CGFloat(pixels) - h) / 2, width: w, height: h))
    }
    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

for size in [16, 32, 128, 256, 512] {
    try render(size).write(to: folder.appendingPathComponent("icon_\(size)x\(size).png"))
    try render(size * 2).write(to: folder.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
}
