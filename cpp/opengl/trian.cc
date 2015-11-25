#include <GL/glut.h>
#include <GL/freeglut.h>
#include <iostream>
#include <math.h>
#include <string>

/**
 * Define
 */
#define SCREEN_TITLE         "Hello World! :D"
#define SCREEN_WIDTH         600
#define SCREEN_HEIGHT         800
#define SCREEN_POSITION_X     700
#define SCREEN_POSITION_Y     10

/**
 * Structure
 */
typedef struct s_badprog
{
    int screenPositionX;
    int screenPositionY;

    double side;
    double hyp;
    double sideHalf;
    double sideToFind;
    double ratio;

    GLclampf bgRed;
    GLclampf bgGreen;
    GLclampf bgBlue;
    GLclampf bgAlpha;

    GLfloat    drawRed;
    GLfloat    drawGreen;
    GLfloat    drawBlue;

    GLfloat z;
    GLfloat x1;
    GLfloat x2;
    GLfloat x3;
    GLfloat y1;
    GLfloat y2;
    GLfloat y3;

} t_badprog;

class BadprogTriangle {
public:
    BadprogTriangle(int *ac, char *av[]);
    virtual ~BadprogTriangle();

    static void initValues(t_badprog *bp);
    static void initProgram(t_badprog *bp);
    static void managerDisplay(void);
    static void managerKeyboard(unsigned char key, int x, int y);
    static void managerResize(int w, int h);
    static void managerMover(int w, int h);
    static void managerClicker(int button, int state, int x, int y);
};
/* BadproG.com */



/**
 * Init values needed in the program
 */
void BadprogTriangle::initValues(t_badprog *bp)
{
    bp->screenPositionX = 100;
    bp->screenPositionY = 100;

    bp->bgRed            = 1;
    bp->bgGreen        = 0;
    bp->bgBlue            = 1;
    bp->bgAlpha        = 0;

    bp->drawRed        = 0;
    bp->drawGreen        = 1;
    bp->drawBlue        = 0;

    bp->side            = 3;
    bp->hyp            = bp->side;
    bp->sideHalf        = bp->hyp / 2;

    bp->sideToFind        = pow(bp->hyp, 2) - pow(bp->sideHalf, 2);
    bp->sideToFind        = sqrt(bp->sideToFind);
    bp->ratio            = pow(bp->sideHalf, 2) / 2;

    bp->z                = -8;

    bp->x1                = -1.5;
    bp->y1                = 0 - bp->ratio;

    bp->x2                = 0;
    bp->y2                = bp->sideToFind - bp->ratio;

    bp->x3                = 1.5;
    bp->y3                = 0 - bp->ratio;
}

/**
 * Call the initValues() function.
 * Set the appropriate background color: glClearColor(pink);
 * Set the appropriate shape color: glColor3f(green);
 * Clear the window to apply the pink color: glClear().
 */
void BadprogTriangle::initProgram(t_badprog *bp)
{
    initValues(bp);
    glClearColor(bp->bgRed, bp->bgGreen, bp->bgBlue, bp->bgAlpha);
    glColor3f(bp->drawRed, bp->drawGreen, bp->drawBlue);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

/**
 * Call iniProgram to have the t_badprog structure set.
 *
 *
 * Drawing a shape inside the glBegin(SHAPE) and glEnd() methods.
 * Each glVertex3f(x, y, z) methods set a vertex.
 * In our case, there are 3 vertices, because we have a triangle.
 * We use glutSwapBuffers() instead of glFlush() because we specified
 * in the main() that we wanted to use double buffering with GLUT_DOUBLE.
 *
 */
void BadprogTriangle::managerDisplay(void)
{
    t_badprog bp;

    initProgram(&bp);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glBegin(GL_TRIANGLES);
        glVertex3f(bp.x1, bp.y1, bp.z);
        glVertex3f(bp.x2, bp.y2, bp.z);
        glVertex3f(bp.x3, bp.y3, bp.z);
    glEnd();
    glutSwapBuffers();
}

/**
 * By clicking ESC key, you close the window
 */
void BadprogTriangle::managerKeyboard(unsigned char key, int x, int y)
{
  std::cout << "key = " << key << std::endl;
  std::cout << "x = " << x << std::endl;
  std::cout << "y = " << y << std::endl;

    switch (key)
    {
        case 27:
        {
            exit(0);
        }
    }
    (void)(x);
    (void)(y);
}

/**
 * By resizing the window, the triangle stays with same proportions.
 */
void BadprogTriangle::managerResize(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w / (double)h, 1.0, 200.0);
}

/**
 * Display in the console the x and y of the cursor when the mouse
 * is moved and a mouse button clicked.
 */
void BadprogTriangle::managerMover(int x, int y)
{
    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl << std::endl;
}

/**
 * Display in the console the x and y of the cursor when a button
 * is clicked, showing its state (0 or 1) and which button is clicked.
 */
void BadprogTriangle::managerClicker(int button, int state, int x, int y)
{

    std::cout << "button = " << button << std::endl;
    std::cout << "state = " << state << std::endl;
    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl << std::endl;
}

/**
 * All init methods needed by openGL.
 * In glutInitDisplayMode() we added GLUT_DOUBLE for double buffering.
 */
BadprogTriangle::BadprogTriangle(int *ac, char *av[]) {
    glutInit(ac,av);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutInitWindowPosition(SCREEN_POSITION_X, SCREEN_POSITION_Y);

    glutCreateWindow(SCREEN_TITLE);

    glutDisplayFunc(BadprogTriangle::managerDisplay);
    glutKeyboardFunc(managerKeyboard);
    glutReshapeFunc(managerResize);
    glutMotionFunc(managerMover);
    glutMouseFunc(managerClicker);
}

// == dtor ==  
BadprogTriangle::~BadprogTriangle() {}

// == main == 
int  main(int ac, char *av[])
{
    BadprogTriangle init(&ac, av);
    glutMainLoop();

    return 0;
}























// == simple kindof hello word == 
/*
#include <GL/glut.h>

void displayMe(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POLYGON);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.5, 0.0, 0.0);
        glVertex3f(0.5, 0.5, 0.0);
        glVertex3f(0.0, 0.5, 0.0);
    glEnd();
    glFlush();
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(300, 300);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Hello world :D");
    glutDisplayFunc(displayMe);
    glutMainLoop();
    return 0;
}
*/
