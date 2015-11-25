
#include <GL/glut.h>
// #include <GL/freeglut.h>



void displayMe() {
  glClear(GL_COLOR_BUFFER_BIT);



  glBegin(GL_POLYGON);
  
    glVertex3f(-1,-1,0.0);
    glVertex3f(1,-1,0.0);
    glVertex3f(0,1,0.0);


  glEnd();

  glFlush();
}


int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(100, 100);
    glutInitWindowPosition(0,0);
    glutCreateWindow("OpenGL window");
    glutDisplayFunc(displayMe);
    

    glClearColor(1,0,0,0); 
    glColor3f(0,1,0); 

    glutMainLoop();
    return 0;
}
