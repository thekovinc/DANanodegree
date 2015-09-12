import csv
import pandas
import time
import scipy.stats
import statsmodels.api as sm
import numpy as np
from ggplot import *
from sklearn.linear_model import SGDRegressor


__author__ = 'Nikolai'

def main(filename):
    #Lesson3_Welch_TTest_Exercise('C:\\Users\\Nikolai\\Downloads\\baseball_stats.csv')
    #Lesson4_lineplot('C:\\Users\\Nikolai\\Downloads\\hr_year.csv')
    #df = pandas.read_csv('C:\\Users\\Nikolai\\Downloads\\turnstile_data_master_with_weather.csv')


    df = pandas.read_csv('C:\\Users\\Nikolai\\Downloads\\turnstile_weather_v2.csv')
    print type(df['ENTRIESn_hourly'])
    print type(df['ENTRIESn_hourly'].values)
    dt = df['datetime']
    print dt[1]
    return 0
    #print type(df.as_matrix(columns='ENTRIESn_hourly'))
    predicts = predictions(df)
    print type(predicts)

    print type(df)
    r2 = Lesson3_Calculating_RSquared(df['ENTRIESn_hourly'].values, predicts)

    print "R^2 = "
    print r2

def Lesson4_Problem2(turnstile_weather):
    turnstile_unit_entries = turnstile_weather[['UNIT', 'ENTRIESn_hourly']]
    turnstile_unit_entries['UNIT'] = turnstile_unit_entries['UNIT'].str.lstrip(to_strip='R')
    print turnstile_unit_entries['UNIT']
    turnstile_by_unit = turnstile_unit_entries.groupby('UNIT', as_index=False).sum()
    plot = ggplot(turnstile_by_unit, aes(x='UNIT', y='ENTRIESn_hourly')) + geom_point() + geom_line() + \
      ggtitle('NYC Subway Entries per Unit') + xlab('Unit') + ylab('Entries')
    print plot

def Lesson4_lineplot_compare(hr_by_team_year_sf_la_csv):
    df = pandas.read_csv(hr_by_team_year_sf_la_csv)
    gg = ggplot(df, aes(x='yearID', y='HR', color='teamID')) + geom_point() + geom_line() + \
      ggtitle('MLB HR hit per year') + xlab('Year') + ylab('HR Hit')
    print gg

def Lesson4_lineplot(hr_year_csv):
    df = pandas.read_csv(hr_year_csv)
    print ggplot(df, aes(x='yearID', y='HR')) + geom_point(color = 'red') + geom_line(color='red') + \
      ggtitle('MLB HR hit per year') + xlab('Year') + ylab('HR Hit') + ylim(0,6000)

def Lesson3_Part3(turnstile_weather):
    rain = turnstile_weather[turnstile_weather['rain']==1]['ENTRIESn_hourly']
    no_rain = turnstile_weather[turnstile_weather['rain']==0]['ENTRIESn_hourly']
    with_rain_mean = np.mean(rain)
    without_rain_mean = np.mean(no_rain)
    U, p = scipy.stats.mannwhitneyu(rain, no_rain)

def Lesson3_Part1(a):
    'C:\\Users\\Nikolai\\PycharmProjects\\UdacityHadoop\\turnstile_data_master_with_weather'

def Lesson3_Calculating_RSquared(data, predictions):
    mean = np.mean(data)
    print "mean type is: " + type(mean).__name__
    print "data type is: " + type(data).__name__
    print "predictions type is: " + type(predictions).__name__
    print "data-predictions type is: " + type(data-predictions).__name__
    mean = np.mean(data)
    r_squared = 1 - np.sum((data-predictions)**2)/np.sum((data-mean)**2)
    #num = np.square(data-predictions, 2)
    #den = np.square(data-mean, 2)
    #r_squared = 1 - np.sum(num)/np.sum(den)
    return r_squared

def Lesson3_Linear_Regression(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    return intercept, params

def Lesson3_Welch_TTest_Exercise(filename):
    df = pandas.read_csv(filename)
    df.fillna(0)
    left = df[df['handedness']=='L']['avg']
    right = df[df['handedness']=='R']['avg']
    result = scipy.stats.ttest_ind(left, right, equal_var=False)
    if result[1] < 0.05:
        bool = False
    else:
        bool = True
    print (bool, result)
    return (bool, result)

def Lesson2_Part11(date):
    date_pieces = time.strptime(date, '%m-%d-%y')
    print time.strftime('%y-%m-%d', date_pieces)

def Lesson2_Part10(time):
    print int(time[:2])

def Lesson2_Part9(filename):
    df = pandas.read_csv(filename, sep = ',')
    df['EXITSn_hourly'] = df['EXITSn'].shift(1)
    df['EXITSn_hourly'] = (df['EXITSn'] - df['EXITSn_hourly']).fillna(0)
    print df[1:1000]

def Lesson2_Part8(filename):
    df = pandas.read_csv(filename, sep = ',')
    df['ENTRIESn_hourly'] = df['ENTRIESn'].shift(1)
    df['ENTRIESn_hourly'] = (df['ENTRIESn'] - df['ENTRIESn_hourly']).fillna(1)
    print df[1:1000]

def Lesson2_Part7(filename):
    df = pandas.read_csv(filename, sep = ',')
    print df[df['DESCn']=='REGULAR'].head

def Lesson2_Part6(filenames, output_file):
    with open(output_file, 'w') as master_file:
       master_file.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')
       reader_out = csv.writer(master_file, delimiter=',')
       for filename in filenames:
            f_in = open(filename, 'r')
            reader_in = csv.reader(f_in, delimiter = ',')
            for line in reader_in:
                reader_out.writerow(line)

def Lesson2_Part5(filename):
    # Open files for reading and writing
    f_in = open(filename, 'r')
    f_out = open('updated_' + filename, 'w')
    reader_in = csv.reader(f_in, delimiter=',')
    reader_out = csv.writer(f_out, delimiter=',')

    # Reads a line from the file
    for line in reader_in:
        col_1 = line[0]
        col_2 = line[1]
        col_3 = line[2]

        # Splits each line of the file into multiple lines
        for i in range(0,(len(line)-3)/5):
            line_for = [col_1, col_2, col_3, line[3+5*i], line[4+5*i], line[5+5*i], line[6+5*i], line[7+5*i].strip()]
            reader_out.writerow(line_for)

    # Closes the files
    f_in.close()
    f_out.close()

def normalize_features(features):
    '''
    Returns the means and standard deviations of the given features, along with a normalized feature
    matrix.
    '''
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features

def recover_params(means, std_devs, norm_intercept, norm_params):
    '''
    Recovers the weights for a linear model given parameters that were fitted using
    normalized features. Takes the means and standard deviations of the original
    features, along with the intercept and parameters computed using the normalized
    features, and returns the intercept and parameters that correspond to the original
    features.
    '''
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.
    """

    clf = SGDRegressor(n_iter = 1000)
    results = clf.fit(features, values)

    intercept = results.intercept_
    params = results.coef_
    ###########################
    ### YOUR CODE GOES HERE ###
    ###########################

    return intercept, params

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.

    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe.
    We recommend that you don't use the EXITSn_hourly feature as an input to the
    linear model because we cannot use it as a predictor: we cannot use exits
    counts as a way to predict entry counts.

    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subset (~50%) of the data contained in
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with
    this exercise on your own computer, locally.

    If you receive a "server has encountered an error" message, that means you are
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features or fewer iterations.
    '''
    ################################ MODIFY THIS SECTION #####################################
    # Select features. You should modify this section to try different features!             #
    # We've selected rain, precipi, Hour, meantempi, and UNIT (as a dummy) to start you off. #
    # See this page for more info about dummy variables:                                     #
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html          #
    ##########################################################################################
    #features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']]
    #features = dataframe[['rain', 'precipi', 'Hour', 'meanpressurei', 'meantempi', 'fog']]
    features = dataframe[['rain', 'precipi', 'hour', 'meanpressurei', 'meantempi', 'fog', 'day_week', 'weekday', 'wspdi', 'tempi', 'pressurei']]
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)

    # Values
    values = dataframe['ENTRIESn_hourly']

    # Get numpy arrays
    features_array = features.values
    values_array = values.values

    means, std_devs, normalized_features_array = normalize_features(features_array)

    # Perform gradient descent
    norm_intercept, norm_params = linear_regression(normalized_features_array, values_array)

    intercept, params = recover_params(means, std_devs, norm_intercept, norm_params)

    predictions = intercept + np.dot(features_array, params)
    # The following line would be equivalent:
    # predictions = norm_intercept + np.dot(normalized_features_array, norm_params)

    return predictions



main('L2P7.txt')
#main('turnstile_110528.txt')