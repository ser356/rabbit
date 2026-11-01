import SwiftUI

// Controles de cristal propios: en la barra del sistema los botones no salen con Liquid Glass.

/// Selector de página: cápsula de cristal con la opción elegida resaltada.
struct PagePicker: View {
    @Binding var page: Page
    @Namespace private var selection

    var body: some View {
        HStack(spacing: 2) {
            ForEach(Page.allCases) { option in
                Button {
                    withAnimation(.smooth(duration: 0.3)) { page = option }
                } label: {
                    Text(option.rawValue)
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(page == option ? Color.white : Color.white.opacity(0.6))
                        .padding(.horizontal, 18)
                        .padding(.vertical, 6)
                        .background {
                            if page == option {
                                Capsule().fill(Color.white.opacity(0.18))
                                    .matchedGeometryEffect(id: "elegida", in: selection)
                            }
                        }
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .glassSurface(Capsule(), interactive: true)
    }
}

/// Botón de cristal dibujado por la app (los de la barra del sistema no respetan el estilo).
struct GlassButton: View {
    enum Size { case small, regular, large }

    let title: String
    let symbol: String
    var tint: Color?
    var size = Size.regular
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: symbol)
                .font(size == .large ? .title3.weight(.semibold) : size == .small ? .callout.weight(.medium)
                      : .body.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, size == .large ? 28 : size == .small ? 14 : 16)
                .padding(.vertical, size == .large ? 14 : size == .small ? 6 : 8)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .glassSurface(Capsule(), tint: tint, interactive: true)
    }
}
