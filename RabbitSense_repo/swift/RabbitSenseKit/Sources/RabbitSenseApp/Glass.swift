import SwiftUI

// Liquid Glass (macOS 26). En sistemas anteriores, materiales translúcidos parecidos.
// El cristal es para la capa que flota sobre el contenido (paneles, controles, etiquetas), no para el contenido.

extension View {
    /// Superficie de cristal con la forma dada, opcionalmente teñida e interactiva.
    @ViewBuilder
    func glassSurface<S: Shape>(_ shape: S, tint: Color? = nil, interactive: Bool = false) -> some View {
        if #available(macOS 26, *) {
            self.glassEffect(Glass.regular.tint(tint).interactive(interactive), in: shape)
        } else {
            self.background(shape.fill((tint ?? .clear).opacity(0.7)))
                .background(.regularMaterial, in: shape)
                .overlay(shape.stroke(Color.white.opacity(0.12), lineWidth: 1))
        }
    }

    /// Botón de cristal; `prominent` para la acción principal (teñido con el acento).
    @ViewBuilder
    func glassButton(prominent: Bool = false) -> some View {
        if #available(macOS 26, *) {
            if prominent { self.buttonStyle(.glassProminent) } else { self.buttonStyle(.glass) }
        } else {
            if prominent { self.buttonStyle(.borderedProminent) } else { self.buttonStyle(.bordered) }
        }
    }

    /// Permite arrastrar la ventana desde esta vista (la ventana no tiene barra de título).
    @ViewBuilder
    func draggableWindow() -> some View {
        if #available(macOS 15, *) {
            self.gesture(WindowDragGesture())
        } else {
            self
        }
    }

    /// Junta varias piezas de cristal cercanas para que se fundan entre sí al moverse.
    @ViewBuilder
    func glassGroup(spacing: CGFloat = 12) -> some View {
        if #available(macOS 26, *) {
            GlassEffectContainer(spacing: spacing) { self }
        } else {
            self
        }
    }
}

/// Fondo de la ventana en pausa: oscuro y neutro, con un resplandor tenue. Es lo que se ve a través del cristal.
struct Backdrop: View {
    let tint: Color

    var body: some View {
        ZStack {
            Color(red: 0.05, green: 0.055, blue: 0.065)
            RadialGradient(colors: [tint.opacity(0.16), .clear], center: UnitPoint(x: 0.32, y: 0.42),
                           startRadius: 10, endRadius: 620)
            RadialGradient(colors: [Color.white.opacity(0.05), .clear], center: UnitPoint(x: 0.85, y: 0.05),
                           startRadius: 10, endRadius: 500)
        }
        .ignoresSafeArea()
    }
}
