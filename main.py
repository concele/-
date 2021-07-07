import numpy as np
from flask import Flask,render_template,request,jsonify,send_file
from web.load_data import load_movie_data
from web.svd import recommend,loadExtData2

app=Flask(__name__)
movie,c=load_movie_data()
back_movie=[]
result=[]
r=np.zeros((200,200))
cf_matrix=np.load('cf.npy')
cf_matrix=np.row_stack((cf_matrix,r))
cf_matrix=np.array(cf_matrix)

@app.route('/')
def index():
    return render_template('reconnect.html')

@app.route('/chose_movie')
def choice():

    return jsonify(
        {
            'M_ID0': str(c[0][0]), 'TITLE0': str(c[0][1]), 'GENRES0': str(c[0][2]),
            'M_ID1': str(c[1][0]), 'TITLE1': str(c[1][1]), 'GENRES1': str(c[1][2]),
            'M_ID2': str(c[2][0]), 'TITLE2': str(c[2][1]), 'GENRES2': str(c[2][2]),
            'M_ID3': str(c[3][0]), 'TITLE3': str(c[3][1]), 'GENRES3': str(c[3][2]),
            'M_ID4': str(c[4][0]), 'TITLE4': str(c[4][1]), 'GENRES4': str(c[4][2]),
            'M_ID5': str(c[5][0]), 'TITLE5': str(c[5][1]), 'GENRES5': str(c[5][2]),
            'M_ID6': str(c[6][0]), 'TITLE6': str(c[6][1]), 'GENRES6': str(c[6][2]),
            'M_ID7': str(c[7][0]), 'TITLE7': str(c[7][1]), 'GENRES7': str(c[7][2]),
            'M_ID8': str(c[8][0]), 'TITLE8': str(c[8][1]), 'GENRES8': str(c[8][2]),
            'M_ID9': str(c[9][0]), 'TITLE9': str(c[9][1]), 'GENRES9': str(c[9][2]),
        }
    )




@app.route('/result',methods=['POST','GET'])
def get_data():
    results=request.form.getlist('data[]')
    for i in range(0,len(results)):
        result.append(results[i])
    return render_template('result.html')


@app.route('/back')
def backdata():
    for i in range(0, len(result)):
        cf_matrix[-1][int(result[i]) - 1] = 5

    back = recommend(cf_matrix, -1)

    for i in range(0, 3):
        back_movie.append(movie.iloc[back[i][0], :])
    return jsonify(
        {
            'M_ID0': str(back_movie[0][0]), 'TITLE0': str(back_movie[0][1]), 'GENRES0': str(back_movie[0][2]),
            'M_ID1': str(back_movie[1][0]), 'TITLE1': str(back_movie[1][1]), 'GENRES1': str(back_movie[1][2]),
            'M_ID2': str(back_movie[2][0]), 'TITLE2': str(back_movie[2][1]), 'GENRES2': str(back_movie[2][2]),
        }
    )

@app.route('/fin')
def fin():
    return render_template('result.html')



if __name__=='__main__':
    app.run()