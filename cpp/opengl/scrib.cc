#include <GL/glut.h>
#include <GL/freeglut.h>
#include <vector> 
#include <utility>
#include <iostream> 

std::vector<std::pair<int,int> > points; 
const int WIDTH = 600; 
const int HEIGHT = 600; 
const double W2 = WIDTH / 2.0; 
const double H2 = HEIGHT / 2.0; 


void displayMe() {
  glClear(GL_COLOR_BUFFER_BIT);



  glBegin(GL_POLYGON);
  
  for(std::size_t i=0; i<points.size(); ++i) {
    //std::cout << "points[" << i << "]= (" << points[i].first << ',' <<  points[i].second << ")\n"; 
    double x = (points[i].first - W2) / W2; 
    double y = (H2 - points[i].second) / H2; 
    glVertex3f(x,y,0.0);
  }

  glEnd();

  glFlush();
}

void handleClick(int button, int state, int x, int y) {

  if(state == 0)
    points.push_back(std::pair<int,int>(x,y)); 

  displayMe(); 
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(0,0);
    glutCreateWindow("Hello world :D");
    glutDisplayFunc(displayMe);
    
    glutMouseFunc(handleClick);

    glClearColor(1,0,0,0); 
    glColor3f(0,1,0); 

    glutMainLoop();
    return 0;
}
