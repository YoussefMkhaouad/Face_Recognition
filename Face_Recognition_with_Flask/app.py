from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)

@app.route('/exec')
def parse(name=None):
    import detect
    print("done")
    return render_template('index.html',name=name)

@app.route('/exec1')
def parse1(name=None):
	import recognizer
	print("done")
	return render_template('index.html',name=name)
	


if __name__ == '__main__':
    app.run()
    app.debug = True