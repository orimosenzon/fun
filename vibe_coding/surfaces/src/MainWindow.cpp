#include "MainWindow.h"
#include "Derivation.h"
#include "DerivationView3D.h"
#include "DiagramScene.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QWidget>
#include <QGroupBox>
#include <QStatusBar>
#include <QSplitter>
#include <QTabWidget>
#include <QScrollArea>
#include <QPushButton>

namespace {
QString descriptionFor(const QString& name) {
    if (name == "Sphere")
        return QStringLiteral(
            "f(u,v) = (sin u · cos v,  sin u · sin v,  cos u)\n"
            "u ∈ [0, π],   v ∈ [0, 2π]\n\n"
            "u — קו אורך מהקוטב הצפוני לדרומי.\n"
            "v — קו רוחב סביב הציר.");
    if (name == "Torus")
        return QStringLiteral(
            "f(u,v) = ((R + r·cos v)·cos u,  (R + r·cos v)·sin u,  r·sin v)\n"
            "u, v ∈ [0, 2π]\n\n"
            "u — תנועה לאורך הצינור הראשי.\n"
            "v — סיבוב סביב חתך הצינור.");
    if (name == "3D Helix (tube)")
        return QStringLiteral(
            "עקומת מרכז: c(u) = (R cos u,  R sin u,  p·u)\n"
            "המשטח הוא הצינור סביב הספירלה (ניבוטר Frenet).\n"
            "u — מיקום לאורך הספירלה.\n"
            "v — סיבוב סביב חתך הצינור.");
    if (name == "Conical spiral")
        return QStringLiteral(
            "ספירלה שרדיוסה מתכווץ ושעולה במעלה.\n"
            "u לאורך העקומה, v סביב חתך הצינור.");
    if (name == "Mobius strip")
        return QStringLiteral(
            "f(u,v) = ((1 + v·cos(u/2))·cos u,  (1 + v·cos(u/2))·sin u,  v·sin(u/2))\n"
            "u ∈ [0, 2π],   v ∈ [-0.4, 0.4]\n\n"
            "משטח חד-צדדי — נסה לעקוב אחרי גוון הצבע.");
    if (name == "Klein bottle")
        return QStringLiteral(
            "אימוס של בקבוק קליין ב-R³ — ללא שפה, ללא 'בפנים'.\n"
            "מוצג כשני חלקים שמתחברים.");
    if (name == "Saddle (z = u*v)")
        return QStringLiteral(
            "f(u,v) = (u, v, u·v)\n"
            "פרבולואיד היפרבולי — דוגמה קלאסית למשטח עם עיקול גאוסי שלילי.");
    return QString();
}
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(tr("Surfaces — parametric surfaces in R³"));
    surfaces_ = allSurfaces();
    buildUi();
    if (!surfaces_.isEmpty()) {
        list_->setCurrentRow(0);
    }
    resize(1100, 720);
}

void MainWindow::buildUi() {
    auto* central = new QWidget(this);
    auto* root = new QHBoxLayout(central);

    // Left panel
    auto* leftBox = new QWidget;
    auto* leftLayout = new QVBoxLayout(leftBox);
    leftLayout->setContentsMargins(8, 8, 8, 8);

    auto* listGroup = new QGroupBox(tr("Surfaces"));
    auto* listLayout = new QVBoxLayout(listGroup);
    list_ = new QListWidget;
    for (const auto& s : surfaces_) {
        list_->addItem(s->name());
    }
    listLayout->addWidget(list_);

    auto* tabs = new QTabWidget;

    info_ = new QLabel;
    info_->setWordWrap(true);
    info_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    info_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    info_->setStyleSheet("QLabel { font-family: monospace; padding: 8px; }");
    auto* defBox = new QWidget;
    auto* defLayout = new QVBoxLayout(defBox);
    defLayout->setContentsMargins(0, 0, 0, 0);
    defLayout->addWidget(info_, 1);
    tabs->addTab(defBox, tr("Definition"));

    // Derivation tab: interactive 3D diagram on top, text below.
    derivation3d_ = new DerivationView3D;
    derivation3d_->setMinimumHeight(300);
    derivation3d_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    auto* resetDiagBtn = new QPushButton(tr("איפוס מבט"));
    auto* topRow = new QWidget;
    auto* topRowL = new QHBoxLayout(topRow);
    topRowL->setContentsMargins(4, 4, 4, 0);
    topRowL->addStretch(1);
    topRowL->addWidget(resetDiagBtn);

    derivationText_ = new QTextBrowser;
    derivationText_->setOpenExternalLinks(false);
    derivationText_->setStyleSheet(
        "QTextBrowser { background:#1b1c20; color:#e6e6ea; "
        "padding:8px; selection-background-color:#3a3a44; "
        "border: none; }");
    derivationText_->setMinimumHeight(180);

    auto* derivBox = new QWidget;
    auto* derivLayout = new QVBoxLayout(derivBox);
    derivLayout->setContentsMargins(0, 0, 0, 0);
    derivLayout->setSpacing(0);
    derivLayout->addWidget(topRow, 0);
    derivLayout->addWidget(derivation3d_, 5);
    derivLayout->addWidget(derivationText_, 4);
    tabs->addTab(derivBox, tr("How it's derived"));

    connect(resetDiagBtn, &QPushButton::clicked,
            derivation3d_, &DerivationView3D::resetView);

    auto* controlsGroup = new QGroupBox(tr("Controls"));
    auto* controlsLayout = new QVBoxLayout(controlsGroup);
    autoRotate_ = new QCheckBox(tr("Auto-rotate"));
    autoRotate_->setChecked(true);
    auto* resetBtn = new QPushButton(tr("Reset view"));
    auto* tip = new QLabel(tr("Drag — orbit  ·  Wheel — zoom"));
    tip->setStyleSheet("color: #888;");
    controlsLayout->addWidget(autoRotate_);
    controlsLayout->addWidget(resetBtn);
    controlsLayout->addWidget(tip);

    leftLayout->addWidget(listGroup, 2);
    leftLayout->addWidget(tabs, 5);
    leftLayout->addWidget(controlsGroup, 0);
    leftBox->setMinimumWidth(380);
    leftBox->setMaximumWidth(460);

    view_ = new SurfaceView(this);

    root->addWidget(leftBox, 0);
    root->addWidget(view_, 1);

    setCentralWidget(central);

    connect(list_, &QListWidget::currentRowChanged,
            this,  &MainWindow::onSurfaceChanged);
    connect(autoRotate_, &QCheckBox::toggled,
            view_, &SurfaceView::setAutoRotate);
    connect(resetBtn, &QPushButton::clicked,
            view_, &SurfaceView::resetView);

    statusBar()->showMessage(
        tr("A surface is a map f: R² → R³ — pick one on the left."));
}

void MainWindow::onSurfaceChanged(int row) {
    if (row < 0 || row >= surfaces_.size()) return;
    auto s = surfaces_[row];
    view_->setSurface(s);
    updateInfo(*s);
}

void MainWindow::updateInfo(const Surface& s) {
    info_->setText(descriptionFor(s.name()));
    QString text = derivationTextFor(s.name());
    if (text.isEmpty()) text = QStringLiteral("<p>(no derivation written)</p>");
    derivationText_->setHtml(text);
    derivation3d_->setScene(diag::sceneFor(s.name()));
}
