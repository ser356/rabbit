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
            self.background(.regularMaterial, in: shape)
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

/// Fondo de la ventana: oscuro, con un resplandor del color del acto actual que se mueve despacio. Es lo que se ve a
/// través del cristal.
struct Backdrop: View {
    let tint: Color

    var body: some View {
        ZStack {
            Color(red: 0.06, green: 0.07, blue: 0.09)
            RadialGradient(colors: [tint.opacity(0.55), .clear], center: UnitPoint(x: 0.2, y: 0.15),
                           startRadius: 20, endRadius: 700)
            RadialGradient(colors: [Color.indigo.opacity(0.45), .clear], center: UnitPoint(x: 0.95, y: 0.9),
                           startRadius: 20, endRadius: 650)
            RadialGradient(colors: [tint.opacity(0.25), .clear], center: UnitPoint(x: 0.75, y: 0.1),
                           startRadius: 10, endRadius: 400)
        }
        .animation(.easeInOut(duration: 1.2), value: tint)
        .ignoresSafeArea()
    }
}
