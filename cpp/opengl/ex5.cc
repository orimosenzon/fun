/*Created By: Cory Lee Kelly */
/*CS 435 Project 2 Scribble */

/* This program provides a simple drawing environment for users allowing
   the drawing of lines, changing of line color, changing of line width, 
   the ability to erase a single line, and the ability to clear all lines
   Also a line can be moved by holding down the select key while selecting a line segment
   (clicking one of the vertices) and dragging the mouse
*/


#include <stdlib.h>
#include <stdio.h>
#include <GL/glut.h>
#include <iostream>

using namespace std;

#define MAX_STROKES 200	/*maximum number of strokes*/
#define MAX_VERTICES 25000


void myMouse(int,int, int, int);
void myMotion(int, int);
void myDisplay();
void myReshape(int, int);
void linewidth_menu(int);
void color_menu(int);
void main_menu(int);

void myinit();

GLsizei wh = 500, ww = 500;

GLfloat width[3] = {1,2,3};
GLfloat colors[8][3]={{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},{0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0},
    {1.0, 1.0, 1.0}};

int present_color = 0;/* default color */
int present_width = 0;/*default width */
int instroke = -1;
int test = 1;
int m;				   /*tells whether or not the shift key is selected*/
int s;
int erasex;
int erasey;
bool selecting = false;/*tells whether a line segment is being selected in order to be erased*/
bool moving = false;/*tells whether a line segment is being selected in order to be moved*/

typedef struct stroke
{ 
     int color; /* color index */
	 int lwidth;/* line width */
     bool used; /* TRUE if stroke exists */
     float x[MAX_VERTICES];		
     float y[MAX_VERTICES];
	 int nvertices;/*number of vertices, should never be greater than two*/
} stroke;

stroke strokes[MAX_STROKES];


void myinit()
{
        int  i;

/* set clear color to white */

        glClearColor(0.5, 0.5, 0.5, 1.0);

/* mark all strokes unused */

        for(i = 0; i<MAX_STROKES; i++) strokes[i].used = false;
}


void myReshape(int w, int h)
{

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity(); 
        gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity(); 

        glViewport(0,0,w,h);

        ww = w;
        wh = h; 
}
/* Changes the present color as well as the next drawn line color to the selected color*/
void color_menu(int index)
{
	int i;
	for(i = 0; i<MAX_STROKES; i++) 
	 {
		if(strokes[i].used == false)
		{
			strokes[i].used = true;
			instroke = i;
			i = MAX_STROKES;
		 }
	 }
	strokes[instroke].color = index;
    present_color =  index;
	
}
/*Changes the present width as well as the next drawn line width to the selected width*/
void linewidth_menu(int index)
{
	strokes[instroke+1].lwidth = index;
    present_width =  index;
}

void main_menu(int index)
{
   int i;
   int j;
   switch(index)
   {
	   case(1)://erase
		   {
		    erasey = wh-erasey;
			selecting = true;
			int i,j,k;
			instroke = -1;
			for(i = 0; i<MAX_STROKES; i++) 
			 {
			  for(j=0;j<strokes[i].nvertices;j++)
				{
				if(strokes[i].x[j] <= (erasex+15) && strokes[i].x[j] >= (erasex-15) &&
						 strokes[i].y[j] <= (erasey+15) && strokes[i].y[j] >= (erasey-15)) 
					 {
						 instroke = i; 
						 i = MAX_STROKES;
						 j = MAX_VERTICES;
						 for(k =0;k<MAX_STROKES;k++){strokes[instroke].x[k] = 0;strokes[instroke].y[k]=0;}
						 strokes[instroke].used = false;
						 strokes[instroke].nvertices = 0;
						 strokes[instroke].color = present_color;
						 strokes[instroke].lwidth = present_width;
						 instroke=-1;				
					  }
			  
			  }
			}
			 
			 glutPostRedisplay();
			 selecting = false;
			 break;
           }
	   case(2)://change color
		   {
			   break;
		   }
	   case(3)://change width
		   {
			   break;
		   }
	   case(4)://clear all lines
		   {
			   for(i = 0; i<MAX_STROKES; i++)
			   {
				   strokes[i].used = false;
				   strokes[i].nvertices = 0;
				   strokes[i].color = 0;
			   for(j=0; j<2; j++) 
			   {
				   strokes[i].x[j]=0; strokes[i].y[j]=0;
			   }
			   }
			   instroke = -1;
			   present_color = 0;
			   present_width = 0;
			   glutPostRedisplay();
			   break;
		   }
	   case(5)://exit the program
		   {
			   exit(0);
			   break;
		   }
   }

}
void menuCheck(int status,int x, int y){
	erasex = x;
	erasey = y;
}

/*This function sets the vertices of the lines based on the different mouse functionalities*/
void myMouse(int btn, int state, int x, int y)
{
    int i,j;
    y = wh-y;
	m = glutGetModifiers();
	
	//A single press of the left mouse button
    if(btn==GLUT_LEFT_BUTTON && state==GLUT_DOWN && m!=GLUT_ACTIVE_SHIFT && !selecting && s != GLUT_MENU_IN_USE )  
    {  
		for(i = 0; i<MAX_STROKES; i++) 
			 {
				 if(strokes[i].used == false)
				 {
					 strokes[i].used = true;
					 instroke = i;
					 i = MAX_STROKES;
				 }
			 }
		i = strokes[instroke].nvertices;
        strokes[instroke].x[i] = x;
		strokes[instroke].y[i] = y;
		strokes[instroke].nvertices++;
		strokes[instroke].color = present_color;
		strokes[instroke].lwidth = present_width;
    }

	//A single relase of the left mouse button
	if(btn==GLUT_LEFT_BUTTON && state==GLUT_UP && m!=GLUT_ACTIVE_SHIFT && !selecting && strokes[instroke].nvertices!=0 && s != GLUT_MENU_IN_USE)  
    {  
		for(i = 0; i<MAX_STROKES; i++) 
			 {
				 if(strokes[i].used == false)
				 {
					 strokes[i].used = true;
					 instroke = i;
					 i = MAX_STROKES;
				 }

			 }
        strokes[instroke].x[test] = x;
		strokes[instroke].y[test] = y;
		strokes[instroke].nvertices++;
		glutPostRedisplay();
		moving = false;
		test=1;
    }
	//A press of the left mouse button while the shift key is being held as well
	  if(btn==GLUT_LEFT_BUTTON && state==GLUT_DOWN && m==GLUT_ACTIVE_SHIFT && s != GLUT_MENU_IN_USE)  
		{
			instroke = -1;
			for(i = 0; i<MAX_STROKES; i++) 
			 {
			  for(j=0;j<strokes[i].nvertices;j++)
				{
				if(strokes[i].x[j] <= (x+3) && strokes[i].x[j] >= (x-3) &&
						 strokes[i].y[j] <= (y+3) && strokes[i].y[j] >= (y-3)) 
					 {
						 instroke = i; 
						 i = MAX_STROKES;
						 j = MAX_VERTICES;
						 moving = true;
					  }
			  
			  }
			}

		}

   
}

void myMotion(int x, int y)
{
	if(m!=GLUT_ACTIVE_SHIFT && selecting == false && s != GLUT_MENU_IN_USE && strokes[instroke].x[0]!=0 && strokes[instroke].y[0]!=0)//drawing line
	{
	y = wh-y;
	int i = 1;
        strokes[instroke].x[test] = x;
		strokes[instroke].y[test] = y;
		strokes[instroke].x[test+1] = x;
		strokes[instroke].y[test+1] = y;
		test+=2;
		strokes[instroke].nvertices+=2;
		glutPostRedisplay();
	}

	if(m==GLUT_ACTIVE_SHIFT && selecting == false && s != GLUT_MENU_IN_USE)//moving
	{

		float dx=0, dy=0;
		int j;
		y = wh-y;	
		//Deciding which vertice to move the line in relation to
		for(j=0;j<strokes[instroke].nvertices;j++)
		{
		if(strokes[instroke].x[j] <= (x+3) && strokes[instroke].x[j] >= (x-3) &&
					 strokes[instroke].y[j] <= (y+3) && strokes[instroke].y[j] >= (y-3))
		{
			dx = x - strokes[instroke].x[j];
			dy = y - strokes[instroke].y[j];
		}
		}
		for(j=0; j<strokes[instroke].nvertices; j++) 
		{
			strokes[instroke].x[j]+=dx; strokes[instroke].y[j]+=dy;
		}

		glutPostRedisplay();
		}
}

void myDisplay()
{

    /* display all used strokes */

    int i, j;

    glClear(GL_COLOR_BUFFER_BIT);
    for(i=0; i<MAX_STROKES; i++)
    {
       if(strokes[i].used)
       {
           glColor3fv(colors[strokes[i].color]);
		   glLineWidth(width[strokes[i].lwidth]);
           glBegin(GL_LINES);
           for(j=0; j<strokes[i].nvertices; j++) glVertex2i(strokes[i].x[j], strokes[i].y[j]);
           glEnd();
       }
    }
    glFlush();
}




int main(int argc, char** argv)
{
    int c_menu;
	int lw_menu;
    glutInit(&argc,argv);
    glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(ww, wh);
    glutCreateWindow("Scribble");
    myinit ();
    c_menu = glutCreateMenu(color_menu);
    glutAddMenuEntry("Black",0);
    glutAddMenuEntry("Red",1);
    glutAddMenuEntry("Green",2);
    glutAddMenuEntry("Blue",3);
    glutAddMenuEntry("Cyan",4);
    glutAddMenuEntry("Magenta",5);
    glutAddMenuEntry("Yellow",6);
    glutAddMenuEntry("White",7);

	lw_menu = glutCreateMenu(linewidth_menu);
    glutAddMenuEntry("1",0);
    glutAddMenuEntry("3",1);
    glutAddMenuEntry("5",2);

    glutCreateMenu(main_menu);
    glutAddMenuEntry("Erase", 1);
    glutAddSubMenu("Line Color", c_menu);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
    glutAddSubMenu("Line Width", lw_menu);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
    glutAddMenuEntry("Clear", 4);
    glutAddMenuEntry("Quit",5);

	glutMenuStatusFunc(menuCheck);
    glutDisplayFunc(myDisplay);
    glutReshapeFunc (myReshape); 
    glutMouseFunc (myMouse);
    glutMotionFunc(myMotion);
    glutMainLoop();

}
