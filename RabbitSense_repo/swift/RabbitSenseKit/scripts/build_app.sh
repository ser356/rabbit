#!/bin/zsh
# Construye build/RabbitSense.app (firmada en local, sin cuenta de desarrollador) a partir del paquete.
# Uso, desde swift/RabbitSenseKit:  zsh scripts/build_app.sh
set -euo pipefail
cd "${0:A:h}/.."
package=$PWD
weights="${RABBITSENSE_WEIGHTS:-${package:h:h}/weights/coreml}"

swift build -c release --product RabbitSenseApp
bin=$(swift build -c release --show-bin-path)

app=build/RabbitSense.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$bin/RabbitSenseApp" "$app/Contents/MacOS/RabbitSense"
cp Sources/RabbitSenseKit/Resources/ytactos_modelo.json "$app/Contents/Resources/"

# Icono: la liebre de SF Symbols en blanco sobre negro.
swiftc -O scripts/make_icon.swift -o build/make_icon
rm -rf build/AppIcon.iconset
build/make_icon build/AppIcon.iconset
iconutil -c icns build/AppIcon.iconset -o "$app/Contents/Resources/AppIcon.icns"

cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>RabbitSense</string>
    <key>CFBundleDisplayName</key><string>RabbitSense</string>
    <key>CFBundleExecutable</key><string>RabbitSense</string>
    <key>CFBundleIdentifier</key><string>local.rabbitsense.app</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleShortVersionString</key><string>0.1</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSCameraUsageDescription</key><string>RabbitSense mira a Cinna con la cámara fija. No guarda vídeo: solo el diario de actos.</string>
    <key>NSCameraUseContinuityCameraDeviceType</key><true/>
    <key>RabbitSenseWeights</key><string>$weights</string>
</dict>
</plist>
PLIST

codesign --force --sign - "$app"
touch "$app"   # que Finder y el Dock refresquen el icono
if [[ ! -d "$weights/owlv2.mlpackage" || ! -d "$weights/vjepa2.mlpackage" ]]; then
    print -u2 "Aviso: no encuentro los modelos en $weights (ejecuta antes rabbitsense.coreml convert)."
fi
print "Lista: $package/$app"
