import PySimpleGUIQt as sg
import json
from Technique1_SVD import SVD
from Technique2_TF import TF

while True:
    ans = input("Which colour scheme would you like? (1 = Brown/Yellow, 2 = White/Blue)\n")
    if ans=='1':
        sg.theme('DarkAmber') 
        break
    if ans=='2':
        break
    print('Please enter 1 or 2.')

while True:
    ans = input("Would you like normal font size or large? (1 = Normal, 2 = Large)\n")
    if ans=='1':
        fontSize = 14
        break
    if ans=='2':
        fontSize = 17
        break
    print('Please enter 1 or 2.')

while True:
    ans = input("Which recommender system would you like to use?\n(1 = Post Filtering, Matrix Factorisation,\n 2 = Contextual Modelling, Tensor Factorisation)\n")
    if ans=='1':
        rs = SVD()
        break
    if ans=='2':
        rs = TF()
        break
    print('Please enter 1 or 2.')


num_of_users = rs.U.shape[0]
num_of_items = rs.m
items = {"Song "+str(i):i for i in range(num_of_items)}

with open("GUI_database/users.json", 'r') as f:
    users = json.load(f)



mnths_tx = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Nov', 'Dec')
mnthsd = {mnth:i for i, mnth in enumerate(mnths_tx)}
locs = [-12,-11,-10,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,13]
locs_str = [str(l) if l<0 else '+'+str(l) for l in locs]
locsd = {l:i for i, l in enumerate(locs_str)}

descr = 'Recommendations will be based on which tracks you like/dislike,\nas well as the location the month you input '
desc2 = '.\nThe number of recommendations will be based on your interaction.\nNo feedback -> 3 recommendations,\n1 or 2 songs reviewed -> 5 recommendations,\notherwise 10 recommendations.'

col = [[sg.Button('Button 1', size=(8,1)),] for i in range(10)]

def update_rec_window(n=10):
    for i in range(3, n):
        window['r'+str(i)].update(visible=True)
        window['lkBtn'+str(i)].update(visible=True)
        window['dlkBtn'+str(i)].update(visible=True)
        window['ex'+str(i)].update(visible=True)
    for i in range(n, 10): 
        window['r'+str(i)].update(visible=False)
        window['lkBtn'+str(i)].update(visible=False)
        window['dlkBtn'+str(i)].update(visible=False)
        window['ex'+str(i)].update(visible=False)

fs_ = 10 if fontSize==14 else 12
recommendations_window = [[sg.Txt('', size=(20,0.6), key='r'+str(i)), 
                        sg.Button('like', size=(8,1), pad=(10,2), font=("Helvetica", 13), key='lkBtn'+str(i)),
                        sg.Button('dislike', size=(8,1), pad=(10,2), font=("Helvetica", 13), key='dlkBtn'+str(i)),
                        sg.Txt('', size=(23,0.8), font=("Helvetica", fs_), key='ex'+str(i))] for i in range(10)]

layout = [
            [sg.Text('Type username:', key='usr_tx', size=(14,1), pad=(16,2)), sg.Text('', key='_OUTPUT_'), 
                sg.Input(do_not_clear=True, pad=(15,2), key='_IN_'), 
                sg.Button('Enter', size=(12,1), pad=(16,2), font=("Helvetica", 12), key='etr_btn')],
            [sg.Txt('Location (Time-Zone)', size=(17,1), pad=(17,2), key='loc_tz'), 
                sg.Drop(locs_str, font=("Helvetica", 14), size=(8,1), pad=(12,1), key='loca'), 
                sg.Txt('     Time (Month)', size=(13,1), pad=(13,2), key='time_tx'), 
                sg.Drop(mnths_tx, size=(8,1), font=("Helvetica", 12), pad=(12,1), key='mnth'), sg.Stretch()],
            [sg.Text('', key='name', size=(18,1), pad=(12,2), visible=False)],
            [sg.Button('Generate\nrecommendations', size=(20,1.5), pad=(20,3), key='gre', font=("Helvetica", 14), visible=False), sg.Button('Exit', size=(20,1.5), pad=(20,3), key='ext', font=("Helvetica", 14), visible=False)],
            [sg.Txt(descr, size=(36,3), pad=(38,4), font="Helvetica 12", key='desc')],
            [sg.Column(recommendations_window, key='list', visible=False)],
            [sg.Button('Submit preferences', font=("Helvetica", 14), key='submit', size=(40,1.5), pad=(60,2), visible=False)],

            [sg.Column(col, key='COL', visible=False)],
          ]

window = sg.Window('Song Recommender', resizable=True).Layout(layout).Finalize()
tx_keys = ['usr_tx', '_OUTPUT_', '_IN_', 'etr_btn', 'loc_tz', 'loca', 'time_tx', 'mnth', 'gre', 'ext'] + ['r'+str(i) for i in range(10)]
for k in tx_keys: window[k].update(font=("Helvetica "+str(fontSize)))
window.Element('desc').Update(visible=False)
window.Refresh()
window.Refresh()

window.Size = window.Size

(w, h) = window.GetScreenDimensions()
[ww, wh] = window.Size
window.move(w//2 - ww//2, wh//2)


def userFeedback(liked, disliked, breakdown, context):
    # print('\n\nYou disliked:\n'+str(disliked))
    print('\n\nYou liked:\n'+str(liked))

    if len(liked) > 0 or len(disliked) > 0:
        # update user profile/model
        review = {iname[0]:1 for iname in liked}
        review.update({iname[0]:0 for iname in disliked})

        rs.feedback(user_id, context, review)

    for l in liked: history_liked.append(int(l[0]))
    for l in disliked: history_disliked.append(int(l[0]))
    return rs.recommend(user_id, history_liked+history_disliked, context, breakdown, newUser=False)

events = []
new_user = True
# num_buttons = 2
entered = False
# any_feedback = False
while True:             # Event Loop
    event, values = window.Read()
    print(event, values, window.Size)
    if event not in (None, 'ext'):
        events.append(event)
        # vals.append(values)
        if event=='etr_btn':
            entered = True
            window.Element('etr_btn').Update(visible=False)
            window.Element('_OUTPUT_').Update(visible=False)
            window.Element('_IN_').Update(visible=False)
            window.Element('usr_tx').Update(visible=False)
            # window.Element('loca').Update(visible=False)
            # window.Element('mnth').Update(visible=False)
            # window.Element('loc_tz').Update(visible=False)
            # window.Element('time_tx').Update(visible=False)

            user = values["_IN_"]
            context = [locsd[values['loca']], mnthsd[values['mnth']]]
            usertxt = user
            user_id = num_of_users
            history_liked, history_disliked = [], []
            if user in users:
                new_user = False
                usertxt = "back "+user
                user_id = int(users[user]["id"])
                history_liked, history_disliked = users[user]['ratings']['liked'], users[user]['ratings']['disliked']
                recommendations = rs.recommend(user_id, history_liked+history_disliked, context, None, newUser=new_user)
            else:
                recommendations = rs.recommend(user_id, [], context, None, newUser=new_user)
            window.Element('name').Update("Welcome "+usertxt, font="Helvetica 14", visible=True)
            usrname, location, month = values['_IN_'], values['loca'], values['mnth']
            window.Element('desc').Update(descr+str((location, month))+desc2)
            window.Element('gre').Update(visible=True)
            window.Element('ext').Update(visible=True)

        if event=='gre':
            window.Element('desc').Update(visible=True)
            window.Element('list').Update(visible=True)
            window.Element('submit').Update(visible=True)

            for i, (rec, com, ty) in enumerate(recommendations):
                window['lkBtn'+str(i)].update('like')
                window['dlkBtn'+str(i)].update('dislike')
                window['r'+str(i)].update('Song '+str(rec))
                window['ex'+str(i)].update(com)

        if event=='Submit\npreferences' or event=='submit':
            context = [locsd[values['loca']], mnthsd[values['mnth']]]
            liked = [recommendations[int(e[-1])] for e in set(events) if e[:-1]=='lkBtn']
            disliked = [recommendations[int(e[-1])] for e in set(events) if e[:-1]=='dlkBtn']
            breakdown = {'N':len([0 for l in liked if l[2]=='N']), 'R':len([0 for l in liked if l[2]=='R']), 'L':len([0 for l in liked if l[2]=='L'])}

            events = []
            recommendations = userFeedback(liked, disliked, breakdown, context)
            print('recommendations: '+str(recommendations))
            window.Element('desc').Update(visible=False)
            update_rec_window(n=len(recommendations))
            window.Element('list').Update(visible=False)
            window.Element('submit').Update(visible=False)
            
        if event == 'Hide Buttons':
            window.Element('SLIDER').Update(visible=False)
            window.Element('COL').Update(visible=False)
        elif event == 'Show Buttons':
            window.Element('SLIDER').Update(visible=True)
            window.Element('COL').Update(visible=True)


        window.Refresh()
        window.Refresh()
    else:
        break
window.Close()
if entered:
    if new_user:
        users[user] = {'id':user_id}
        users[user]['ratings'] = {'liked':history_liked, 'disliked':history_disliked}
    else:
        users[user]['ratings']['liked'] += history_liked
        users[user]['ratings']['disliked'] += history_disliked
    with open('GUI_database/users.json', 'w') as outfile:
        outfile.write(json.dumps(users))
    if ans=='1': rs.save_pred()
    print('Saved user preferences')