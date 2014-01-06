to start the web server: 

python manage.py runserver

browse to http://127.0.0.1:8000/ to see the page 


to add entries to the data base: 
python manage.py shell
>>> from chk1.models import myline
>>> l = myline(t = "hello there")
>>> l.save()
>>> exit()
