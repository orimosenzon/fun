#pragma once

#include <QString>
#include <QByteArray>

// Intuitive explanation text (rich text / HTML, no <svg>) for the
// given surface. Empty if unknown.
QString derivationTextFor(const QString& surfaceName);

// Standalone SVG document illustrating angles and distances for the
// given surface. Empty if unknown.
QByteArray derivationSvgFor(const QString& surfaceName);
