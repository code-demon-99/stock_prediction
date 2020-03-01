from django.shortcuts import render,redirect
from django.http import HttpResponse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt , mpld3
from sklearn.preprocessing import scale
from TFANN import ANNR
import os 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

info ={}
context = {}
info ['stocklist'] = ['Reliance','HDFC','UltraCement','Eicher ','MRF'
    ,'Surya Roshni']
file_names={
    'Reliance': 'Data_sets/RELIANCErr2.csv' ,
    'HDFC' : 'Data_sets/HDFC.csv' ,
    'UltraCement'  : 'Data_sets/ULTRACEMCO.BO.csv',
    'Eicher':'Data_sets/EICHERMOT.BO.csv',
    'MRF':'Data_sets/MRF.NS.csv',
    'Surya Roshni': 'Data_sets/SURYAROSNI.BO.csv',

}
def welcome_page(request):
    if request.method == 'POST':
        date_for  = request.POST.get('date')
        stock_selected = request.POST.get('stock_name')
        datetime_object = datetime.strptime(date_for, '%b %d, %Y')
        context['selected_stock'] = stock_selected
        context['Date_selected'] = datetime_object
        return redirect('prediction:page2')
        
    return render(request,'prediction/index.html' ,info)

def page2(request):
    stock_data = np.loadtxt(os.path.join(BASE_DIR,f"prediction/project_api/{file_names[context['selected_stock']]}"), delimiter=",", skiprows=1, usecols=(1, 4))
    stock_data=scale(stock_data)
    prices = stock_data[:, 1].reshape(-1, 1)
    dates = stock_data[:, 0].reshape(-1, 1)
    input = 1
    output = 1
    hidden = 50
    layers = [('F', hidden), ('AF', 'tanh'), ('F', hidden), ('AF', 'tanh'), ('F', hidden), ('AF', 'tanh'), ('F', output)]
    mlpr = ANNR([input], layers, batchSize = 256, maxIter = 20000, tol = 0.2, reg = 1e-4, verbose = True)  
    holdDays = 5
    totalDays = len(dates)
    mlpr.fit(dates[0:(totalDays-holdDays)], prices[0:(totalDays-holdDays)])  
    pricePredict = mlpr.predict(dates)
    fig = plt.figure()
    plt.plot(dates, prices)
    plt.plot(dates, pricePredict, c='#5aa9ab')
    context['graph1'] = mpld3.fig_to_html(fig)
    plt.close()
    return render(request,'prediction/graphs.html',context)