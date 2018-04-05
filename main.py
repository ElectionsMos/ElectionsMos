
# coding: utf-8

# # Сбор результатов выборов 2011, 2012, 2013, 2016 и 2018 годов для каждого УИКа с сайта Мосгоризбиркома

# In[1]:


import pandas as pd
from robobrowser import RoboBrowser


# In[2]:


#Ссылки на страницы Мосгоризбиркома для выборов разных лет 
LINK_2011 = 'http://www.moscow_city.vybory.izbirkom.ru/region/region/moscow_city?action=show&root=1000149&tvd=100100028713457&vrn=100100028713299&region=77&global=true&sub_region=77&prver=0&pronetvd=null&vibid=100100028713457&type=242'
LINK_2012 = 'http://www.moscow_city.vybory.izbirkom.ru/region/region/moscow_city?action=show&root=1000020&tvd=100100031793849&vrn=100100031793505&region=77&global=true&sub_region=77&prver=0&pronetvd=null&vibid=100100031793849&type=226'
LINK_2013 = 'http://www.moscow_city.vybory.izbirkom.ru/region/region/moscow_city?action=show&root=1&tvd=27720001368293&vrn=27720001368289&region=77&global=&sub_region=77&prver=0&pronetvd=null&vibid=27720001368293&type=234'
LINK_2016 = 'http://www.moscow_city.vybory.izbirkom.ru/region/region/moscow_city?action=show&root=1000259&tvd=100100067796113&vrn=100100067795849&region=77&global=1&sub_region=77&prver=0&pronetvd=0&vibid=100100067796113&type=242'
LINK_2018 = 'http://www.moscow_city.vybory.izbirkom.ru/region/region/moscow_city?action=show&root=1000074&tvd=100100084849200&vrn=100100084849062&region=77&global=true&sub_region=77&prver=0&pronetvd=null&vibid=100100084849200&type=226'

#Количество кандидатов
CAND_2011 = 7
CAND_2012 = 5
CAND_2013 = 6
CAND_2016 = 14
CAND_2018 = 8

#Типы выборов(необходимо для корректного скреппинга с сайта Мосгоризбиркома):
MAYOR = [4, 5]
PRES = [5, 6]
PARL = [4, 6]


# In[3]:


def get_page(rb, cand, el_type):
    """
    Функция принимает RoboBrowser с открытой страницей результатов голосования по какому-то УИКу rb,
    кол-во кандидатов в выборах cand, а также тип выборов el_type и возвращает список, состоящий из номера УИКа,
    кол-ва избирателей на УИКе, кол-во проголосовавших на дому и на участке, кол-во испорченных голосов,
    а также кол-во голосов за каждого из избирателей.
    
    """
    uik_result = []
    main_table = rb.parsed.body('table')[2]('tr')[3].td
    uik_result.append(int(main_table("table")[2]('tr')[1]('td')[1].string.split("№")[1]))
    main_table = main_table("table")[el_type[0]]('tr')
    try:
        uik_result.append(int(main_table[0]('td')[2].b.string))
    except IndexError:
        for x in range(4+cand):
            uik_result.append(-1)
        return uik_result
    for m in range(el_type[1],el_type[1]+3):
        uik_result.append(int(main_table[m]('td')[2].b.string))
    for l in range(-cand,0):
        uik_result.append(int(main_table[l]('td')[2].b.string))
    return uik_result


# In[4]:


def get_vote_res(link, cand, el_type):
    """
    Функция принимает на вход ссылку на страницу Мосгоризбиркома для выборов какого-то из годов link, кол-во кандидатов на этих
    выборах cand, а также тип выборов el_type и возвращает pandas dataframe с результатами голосования для каждого УИКа
    (данные из функции get_page).
    """
    voting = pd.DataFrame()
    q = RoboBrowser()
    q.open(link)
    form = q.get_form()
    for i in range(1,len(form['gs'].options)):
        q.open(form['gs'].options[i])
        form_in_dist = q.get_form()
        for j in range(1,len(form_in_dist['gs'].options)):
            q.open(form_in_dist['gs'].options[j])
            if el_type == PARL:
                form_parl = q.get_form()
                for k in range(1,len(form_parl['gs'].options)):
                    q.open(form_parl['gs'].options[k])
                    voting = voting.append([get_page(rb = q, cand = cand, el_type = el_type)], ignore_index=True)
            else:
                voting = voting.append([get_page(rb = q, cand = cand, el_type = el_type)], ignore_index=True)
    return voting


# In[5]:


def voting_data_collect():
    """
    Функция возвращает собирает результаты голосования для каждых выборов,
    делает сооветствующие наименования,
    оставляя только те участки, где кол-во избирателей больше нуля,
    а затем экспортирует их в excel.
    """
    voting_2011 = get_vote_res(link = LINK_2011, num_of_cand = CAND_2011, type_of_el = PARL)
    voting_2011.columns = ["УИК","ИЗБИРАТЕЛЕЙ","ЯЩИК","УРНА","ИСПОРЧЕНО","СПРАВ РОС_АБС","ЛДПР_АБС","ПАТРИОТЫ_АБС","КПРФ_АБС","ЯБЛОКО_АБС", "ЕДИНАЯ РОС_АБС","ПРАВОЕ ДЕЛО_АБС"]
    voting_2011 = voting_2011[voting_2011['ИЗБИРАТЕЛЕЙ'] > 0]
    voting_2012 = get_vote_res(link = LINK_2012, num_of_cand = CAND_2012, type_of_el = PRES)
    voting_2012.columns = ["УИК","ИЗБИРАТЕЛЕЙ","ЯЩИК","УРНА","ИСПОРЧЕНО","ЖИРИНОВСКИЙ_АБС","ЗЮГАНОВ_АБС","МИРОНОВ_АБС","ПРОХОРОВ_АБС","ПУТИН_АБС"]
    voting_2012 = voting_2012[voting_2012['ИЗБИРАТЕЛЕЙ'] > 0]
    voting_2013 = get_vote_res(link = LINK_2013, num_of_cand = CAND_2013, type_of_el = MAYOR)
    voting_2013.columns = ["УИК","ИЗБИРАТЕЛЕЙ","ЯЩИК","УРНА","ИСПОРЧЕНО","ДЕГТЯРЕВ_АБС","ЛЕВИЧЕВ_АБС","МЕЛЬНИКОВ_АБС","МИТРОХИН_АБС","НАВАЛЬНЫЙ_АБС","СОБЯНИН_АБС"]
    voting_2013 = voting_2013[voting_2013['ИЗБИРАТЕЛЕЙ'] > 0]
    voting_2016 = get_vote_res(link = LINK_2016, num_of_cand = CAND_2016, type_of_el = PARL)
    voting_2016.columns = ["УИК","ИЗБИРАТЕЛЕЙ","ЯЩИК","УРНА","ИСПОРЧЕНО","РОДИНА_АБС","КОММУНИСТЫ_АБС","ПЕНСИОНЕРЫ_АБС","ЕДИНАЯ РОС_АБС","ЗЕЛЕНЫЕ_АБС","ГРАЖ ПЛАТФОРМА_АБС","ЛДПР_АБС","ПАРНАС_АБС","ПАРТИЯ РОСТА_АБС","ГРАЖ СИЛА_АБС","ЯБЛОКО_АБС","КПРФ_АБС","ПАТРИОТЫ_АБС","СПРАВ РОС_АБС"]
    voting_2016 = voting_2016[voting_2016['ИЗБИРАТЕЛЕЙ'] > 0]
    voting_2018 = get_vote_res(link = LINK_2018, num_of_cand = CAND_2018, type_of_el = PRES)
    voting_2018.columns = ["УИК","ИЗБИРАТЕЛЕЙ","ЯЩИК","УРНА","ИСПОРЧЕНО","БАБУРИН_АБС","ГРУДИНИН_АБС","ЖИРИНОВСКИЙ_АБС","ПУТИН_АБС","СОБЧАК_АБС","СУРАЙКИН_АБС","ТИТОВ_АБС","ЯВЛИНСКИЙ_АБС"]
    voting_2018 = voting_2018[voting_2018['ИЗБИРАТЕЛЕЙ'] > 0]
    
    voting_2011.to_excel("voting_2011.xlsx")
    voting_2012.to_excel("voting_2012.xlsx")
    voting_2013.to_excel("voting_2013.xlsx")
    voting_2016.to_excel("voting_2016.xlsx")
    voting_2018.to_excel("voting_2018.xlsx")
    return


# # Подготовка данных о результатах голосования

# In[6]:


import numpy as np


# In[7]:


#Инициализируем глобальные массивы с результатами голосования по УИКам. Если они не были собраны до этого, то собираем их.
#try:
VOTING_2011 = pd.read_excel("voting_2011.xlsx")
VOTING_2012 = pd.read_excel("voting_2012.xlsx")
VOTING_2013 = pd.read_excel("voting_2013.xlsx")
VOTING_2016 = pd.read_excel("voting_2016.xlsx")
VOTING_2018 = pd.read_excel("voting_2018.xlsx")
#except FileNotFoundError:
#    voting_data_collect()
#    VOTING_2011 = pd.read_excel("voting_2011.xlsx")
#    VOTING_2012 = pd.read_excel("voting_2012.xlsx")
#    VOTING_2013 = pd.read_excel("voting_2013.xlsx")
#    VOTING_2016 = pd.read_excel("voting_2016.xlsx")
#    VOTING_2018 = pd.read_excel("voting_2018.xlsx")


# In[8]:


YEARS = [2011,2012,2013,2016,2018]


#Вводим дополнительные таблицы с результатами голосования по России, цифры взяты с сайта ЦИК.
RUS_VOTING_2011 = pd.DataFrame([[109237780, 4522236, 61134290, 1033464, 8695522, 7664570, 639119, 12599507, 2252403, 32379135, 392806]],columns = VOTING_2011.columns[1:5+CAND_2011])
RUS_VOTING_2012 = pd.DataFrame([[109860331, 6139277, 65562388, 836691, 4458103, 12318353, 2763935, 5722508, 45602075]],columns = VOTING_2012.columns[1:5+CAND_2012])
RUS_VOTING_2016 = pd.DataFrame([[110061200, 3524522, 49107327, 982596, 792226, 1192595, 910848, 28527828, 399429, 115433, 6917063, 384675, 679030, 73971, 1051335, 7019752, 310015, 3275053]],columns = VOTING_2016.columns[1:5+CAND_2016])
RUS_VOTING_2018 = pd.DataFrame([[109008428, 5039911, 68539081, 791258, 479013, 8659206, 4154985, 56430712, 1238031, 499342, 556801, 769644]],columns = VOTING_2018.columns[1:5+CAND_2018])

#Объявляем, кого на выборах мы считаем кандидатом от власти, а кого --- оппозицией.
CANDIDATES = pd.DataFrame([["ЕДИНАЯ РОС",2011,"ВЛАСТЬ"],["ЯБЛОКО",2011,"ОППОЗИЦИЯ"],["ЕДИНАЯ РОС",2016,"ВЛАСТЬ"],["ЯБЛОКО",2016,"ОППОЗИЦИЯ"],["ПАРНАС",2016,"ОППОЗИЦИЯ"],["ПУТИН",2012,"ВЛАСТЬ"],["ПРОХОРОВ",2012,"ОППОЗИЦИЯ"],["СОБЯНИН",2013,"ВЛАСТЬ"],["НАВАЛЬНЫЙ",2013,"ОППОЗИЦИЯ"],["СОБЧАК",2018,"ОППОЗИЦИЯ"],["ЯВЛИНСКИЙ",2018,"ОППОЗИЦИЯ"],["ПУТИН",2018,"ВЛАСТЬ"]],columns = ["КАНДИДАТ/ПАРТИЯ","ГОД","СТАТУС"])


# In[9]:


def count_the_rel_res(voting, cand, rus = False):
    """
    Функция принимает на вход pandas dataframe с результатами голосования, кол-во кандидатов на выборах,
    а также флаг, являются ли результаты общероссийскими или нет (нет по умолчанию).
    Функция добавляет в каждую таблицу явку, а также относительный результат каждого из кандидатов
    """
    voting['ЯВКА_АБС'] = voting['ЯЩИК'] + voting['УРНА']
    voting['ЯВКА_ОТН'] = voting['ЯВКА_АБС'] / voting['ИЗБИРАТЕЛЕЙ']
    k = 0
    if rus:
        k = 1
    for i in range(cand):
        voting[voting.iloc[:,5+i-k].name[:-3] + "ОТН"] = voting.iloc[:,5+i-k]/voting["ЯВКА_АБС"]
    return

DATE_VOTING = [[2011,VOTING_2011, RUS_VOTING_2011, CAND_2011],[2012,VOTING_2012, RUS_VOTING_2012, CAND_2012],[2013,VOTING_2013, VOTING_2013, CAND_2013],[2016,VOTING_2016,RUS_VOTING_2016, CAND_2016],[2018,VOTING_2018,RUS_VOTING_2018, CAND_2018]]

#Считаем явку и относительные результаты для каждых выборов по Москве и по России (выборы мэра Москвы в 2013 только для Москвы)
for year in DATE_VOTING:
    count_the_rel_res(year[1], year[3])
    if year[0] == 2013:
        continue
    count_the_rel_res(year[2], year[3], rus = True)


# In[10]:


def res_uiks_mos_rus():
    """
    Функция считает для каждого УИКа, а также в целом по Москве и по России
    относительные совокупные результаты голосов за власть, оппозицию, а также явку на каждых из пяти выборов.
    Функция возвращает список из трех таблиц: результатов для УИКов (могут быть np.NaN, если УИКа не было на выборах),
    Москвы и России соответственно.
    """
    mos_res = pd.DataFrame(index = ["ЯВКА","ОПП","ВЛАСТЬ"])
    rus_res = pd.DataFrame(index = ["ЯВКА","ОПП","ВЛАСТЬ"])
    for year in DATE_VOTING:
        base = pd.concat([year[1]["УИК"],year[1]["ЯВКА_ОТН"].rename("ЯВКА_"+str(year[0])),                          year[1][CANDIDATES[(CANDIDATES["ГОД"] == year[0]) & (CANDIDATES["СТАТУС"] == "ОППОЗИЦИЯ")]["КАНДИДАТ/ПАРТИЯ"]                                  .apply(lambda x: x+'_ОТН')].apply(np.sum, axis=1).rename("ОПП_"+str(year[0])),                          year[1][CANDIDATES[(CANDIDATES["ГОД"] == year[0]) & (CANDIDATES["СТАТУС"] == "ВЛАСТЬ")]["КАНДИДАТ/ПАРТИЯ"]                                  .apply(lambda x: x+'_ОТН')].apply(np.sum, axis=1).rename("ВЛАСТЬ_"+str(year[0]))],axis=1)
        weight_p = year[1]["ИЗБИРАТЕЛЕЙ"]/np.sum(year[1]['ИЗБИРАТЕЛЕЙ'])
        weight_t = year[1]["ЯВКА_АБС"]/np.sum(year[1]['ЯВКА_АБС'])
        mos_res[year[0]] = np.append(np.dot(weight_p,base["ЯВКА_"+str(year[0])]),                                     np.dot(weight_t,base[["ОПП_"+str(year[0]),"ВЛАСТЬ_"+str(year[0])]]))
        if year[0] != 2013:
            rus_res[year[0]] = np.array([year[2]["ЯВКА_ОТН"].values[0],                                         year[2][CANDIDATES[(CANDIDATES["ГОД"] == year[0]) & (CANDIDATES["СТАТУС"] == "ОППОЗИЦИЯ")]                                                 ["КАНДИДАТ/ПАРТИЯ"].apply(lambda x: x+'_ОТН')].apply(np.sum, axis=1).values[0],                                         year[2][CANDIDATES[(CANDIDATES["ГОД"] == year[0]) & (CANDIDATES["СТАТУС"] == "ВЛАСТЬ")]                                                 ["КАНДИДАТ/ПАРТИЯ"].apply(lambda x: x+'_ОТН')].apply(np.sum, axis=1).values[0]])
        if year[0] == 2011:
            res_uiks = base
            continue
        res_uiks = res_uiks.merge(base, how="outer")
    return [res_uiks,mos_res,rus_res]


# In[11]:


#Ининциализируем глобальные массивы с относительными совокупными результатами по явке и голосам за власть и оппозицию
#нужны для визуалиции
[UIKS_RESULTS, MOS_RESULTS, RUS_RESULTS] = res_uiks_mos_rus()


# In[12]:


def uik_res_info(uik):
    """
    Функция принимает на вход номер УИКа и возвращает pandas dataframe, с явкой и относительными результатами голосования
    за власть и оппозицию для каждого года (могут быть np.NaN, если данного УИКа не было на выборах).
    В случае, если УИКа с данным номером не было ни на одних выборах, возвращается -1.
    """
    uik_data = UIKS_RESULTS[UIKS_RESULTS["УИК"] == uik]
    if len(uik_data["УИК"]) == 0:
        return -1
    df = pd.DataFrame(index = ["ЯВКА","ОПП","ВЛАСТЬ"])
    for year in YEARS:
        df[year] = [uik_data["ЯВКА_"+str(year)].values[0],uik_data["ОПП_"+str(year)].values[0],uik_data["ВЛАСТЬ_"+str(year)].values[0]]
    return df

def get_year_res(year):
    """
    Функция принимает на вход год, в который проходили выборы (int), возвращает pandas dataframe с явкой и относительными
    результатами голосования за власть и оппозицию по всем УИКам, которые существовали в данный год.
    """
    return UIKS_RESULTS[np.append(np.array("УИК"),UIKS_RESULTS.columns[UIKS_RESULTS.columns.str.contains(str(year))])].dropna()


# # Функции, отвечающие за визуализацию данных по УИКу и для выборов в целом

# In[13]:


import plotly as pl
import plotly.graph_objs as go
with open('PLOTLY_KEY.txt') as f:
    PLOTLY_KEY = f.read().split()[0]
pl.tools.set_credentials_file(username='ElectionsMos', api_key=PLOTLY_KEY)

import seaborn as sb
import matplotlib.pyplot as plt

import random


# In[14]:


def pies_for_uik(uik):
    """
    Функция принимает на вход номер УИКа (int) и рисует круговые диаграммы результатов тех выборов,
    на которых данный УИК был представлен, загружая рисунок под именем 'uik_pies_uik' в формате png
    """
    try:
        num_of_pies = len(uik_res_info(uik).dropna(axis=1).columns.values)
    except AttributeError:
        return -1
    data = []
    annotations = []
    colors = ['red','pink','cyan','silver','orange','olive','yellow','orchid','#00BFFF','#FF1493','wheat','azure','#90EE90','lime','#D2691E']
    pies_domain_donut = {
        1: [{'x': [0,1], 'y': [0.1,1]},
           {'x': .5, 'y': 0.56}],
        2: [{'x': [0,0.5], 'y': [0.15,1]},{'x': [0.5,1], 'y': [0.15,1]},
           {'x': .225, 'y': 0.58},{'x': .775, 'y': 0.58}],
        3: [{'x': [0,0.4], 'y': [0.35,1]},{'x': [0.3,0.7], 'y': [.1,0.75]},{'x': [0.6,1], 'y': [0.35,1]},
           {'x': .175, 'y': 0.71},{'x': .5, 'y': .43},{'x': .825, 'y': 0.71}],
        4: [{'x': [0,0.4], 'y': [0.47,1]},{'x': [0.4,0.8], 'y': [0.47,1]},{'x': [0.2,0.6], 'y': [0.05,0.58]},{'x': [0.6,1], 'y': [0.05,0.58]},
           {'x': .178, 'y': 0.77},{'x': .6, 'y': 0.77},{'x': .4, 'y': 0.29},{'x': .82, 'y': 0.29}],
        5: [{'x': [0,0.4], 'y': [0.53,1]},{'x': [0.15,0.55], 'y': [0.05,0.52]},{'x': [0.3,0.7], 'y': [0.53,1]},{'x': [0.45,0.85], 'y': [0.05,0.52]},{'x': [0.6,1], 'y': [0.53,1]},
           {'x': .182, 'y': 0.8},{'x': .35, 'y': 0.26},{'x': .5, 'y': 0.8},{'x': .667, 'y': 0.26},{'x': .817, 'y': 0.8}]
    }
    uik_domain = pies_domain_donut[num_of_pies]
    pie_num = 0
    for year in DATE_VOTING:
        if len(year[1][year[1]["УИК"] == uik]["УИК"]) == 0:
            continue
        pie_dict = {'type': 'pie', 'hoverinfo':'label+value','textinfo':'label+percent','insidetextfont': {'size': max(12-num_of_pies,8)},'outsidetextfont':{'size': min(max(9-num_of_pies,5),7)}
                   ,'hole': 0.25, 'marker': {'colors': random.sample(colors,year[3]+1), 'line': {'color':'black', 'width':1}}
                   }
        pie_dict['name'] = str(year[0])
        pie_dict['labels'] = [year[1].columns.values[4]] + [x[:-4] for x in year[1].columns.values[5:5+year[3]]]
        pie_dict['values'] = year[1][year[1]["УИК"] == uik][year[1].columns.values[4:5+year[3]]].values[0]
        pie_dict['domain'] = uik_domain[pie_num]
        data.append(pie_dict)
        ann_dict = {"font": {"size": max(13,17-num_of_pies)},"showarrow": False,"text": str(year[0])}
        ann_dict.update(uik_domain[pie_num+num_of_pies])
        annotations.append(ann_dict)
        pie_num += 1
    figure = {'data': data, 'layout': {'title': 'УИК №'+str(uik), 'annotations': annotations, 'showlegend': False}}
    pl.plotly.image.save_as(figure, filename='uik_pies_'+str(uik), format='png',width = 1141)
    return


# In[15]:


def collect_dyn_data(uik):
    """
    Функция принимает на вход номер УИКа (int) и возвращает список с данными для рисования time_series данных.
    Нулевой элемент --- номер УИКа, оставшиеся девять элементов --- данные для рисования time_series графиков
    для УИКа, Москвы и РФ сначала явки (три элемента), затем голосов за оппозицию (три элемента, графики пунктиром),
    затем голосов за власть (три элемента). Если УИКа с таким номером нет в базе данных, то возвращается -1. 
    """
    data = [uik]
    for feature in MOS_RESULTS.index.values:
        line_dict = {'color': '#20B2AA', 'width': 4}
        if feature == "ОПП":
            line_dict['dash'] = 'dash'
        try:
            data.append(
                go.Scatter(x=uik_res_info(uik=uik).dropna(axis=1).loc[feature].index,
                           y=uik_res_info(uik=uik).dropna(axis=1).loc[feature].values,
                           name = "УИК №"+str(uik)+' '+feature,
                           marker = {'symbol': 'star-diamond', 'size': 10},
                           line = line_dict))
        except AttributeError:
            return -1
        
        line_dict = {'color': 'red'}
        if feature == "ОПП":
            line_dict['dash'] = 'dash'
        data.append(go.Scatter(
        x=MOS_RESULTS.loc[feature].index,
        y=MOS_RESULTS.loc[feature].values,
        name = "МОСКВА"+' '+feature,
        marker = {'symbol': 'star', 'size': 10},
        line = line_dict,
        opacity = 0.7))
        
        line_dict = {'color': 'green'}
        if feature == "ОПП":
            line_dict['dash'] = 'dash'
        data.append(go.Scatter(
        x=RUS_RESULTS.loc[feature].index,
        y=RUS_RESULTS.loc[feature].values,
        name = "РФ"+' '+feature,
        marker = {'symbol': 'pentagon', 'size': 10},
        line = line_dict,
        opacity = 0.7))
    return data


# In[16]:


def dynamics(data, turnout = True):
    """
    Функция принимает на вход данные для рисования time_series графиков, возвращаемых функцией collect_dyn_data(uik),
    и рисует графики, загружая рисунок под именем 'turnout_uik', "powopp_uik" для динамики явки и
    доли голосов за власть/оппозицию соответственно в формате png. 
    Если флаг turnout=True, то сохраняется только динамика явки (по умолчанию),
    если False, то только доля голосов за власть/оппозицию
    Если данных передано не было (отсутствие УИКа в базе данных), то возвращается -1.
    """
    if data == -1:
        return -1
    layout = {'yaxis': {'range': [0,1]}}
    if turnout:
        layout['title']= "Динамика явки"
        figure = {'data':data[1:4], 'layout':layout}
        pl.plotly.image.save_as(figure, filename = "turnout_"+str(data[0]), format='png',width = 1141)
        return
    layout['title']= "Доля голосов за власть/оппозицию"
    figure = {'data':data[4:], 'layout':layout}
    pl.plotly.image.save_as(figure, filename = "powopp_"+str(data[0]), format='png',width = 1141)
    return


# In[17]:


def get_distributions():
    """
    Функция рисует и сохраняет в текущий каталог распределения явки, доли голосов за власть и за оппозицию для каждых выборов,
    используя каждый УИК в качестве одного наблюдения. Файл называется 'distributions.png'.
    """
    figure, axes = plt.subplots(nrows = 3, ncols = 5, figsize=(12, 7),sharex=True)
    sb.set(color_codes=True)
    sb.despine(left=True)
    for i, year in enumerate(YEARS):
        sb.distplot(get_year_res(year)['ЯВКА_'+str(year)],ax= axes[0,i])
        sb.distplot(get_year_res(year)['ВЛАСТЬ_'+str(year)],ax= axes[1,i])
        sb.distplot(get_year_res(year)['ОПП_'+str(year)],ax= axes[2,i])
    ###FROM http://seaborn.pydata.org/examples/distplot_options.html#distplot-options
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    ###END FROM
    figure.savefig('./distributions.png', orientation='landscape')
    return


# # Манипуляция с файлами 

# In[18]:


import os


# In[19]:


FILENAMES = ["uik_pies_","turnout_","powopp_"]


# In[20]:


def delete_files(uik):
    """
    Функция принимает номер УИК и удаляет графики об УИКе оттуда.
    """
    for filename in FILENAMES:
        pic = filename+str(uik)+'.png'
        if os.path.exists(pic):
            os.remove(pic)
    return


# # Функции обработки строк и скреппинга с помощью Selenium

# In[21]:


from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep


# In[22]:


CIK_URL = "http://cikrf.ru/services/lk_address?do=address"
DIGITS = {'0','1','2','3','4','5','6','7','8','9'}


# In[23]:


def district_str_form(name):
    """
    Функция принимает на вход текстовую строку name и проводит ее к виду, в котором названия районов написаны на сайте ЦИК
    (каждое отдельное слово, а также части слова через дефис с заглавный буквы, остальные строчные).
    Возвращает строчку в новом виде.
    """
    good_str = []
    for word in name.split():
        if len(word) > 1:
            good_substr = []
            for subword in word.split('-'):
                if len(subword) > 1:
                    good_substr.append(subword[0].upper() + subword.lower()[1:])
                else:
                    good_substr.append(subword.upper())
            good_str.append('-'.join(good_substr))    
        else:
            good_str.append(word.upper())
    return ' '.join(good_str)


# In[24]:


def intersect_list(list_of_lists):
    """
    Функция принимает на вход список списков и вовращает список из элементов в пересечении этих списков,
    а если пересечение пусто, то их объединение.
    """
    main_set = set(list_of_lists[0])
    for a_list in list_of_lists:
        main_set.intersection_update(set(a_list))
    if len(main_set) > 0:
        return list(main_set)
    #объединяем, если пересечение пусто
    main_set = set(list_of_lists[0])
    for a_list in list_of_lists:
        main_set.update(set(a_list))
    return list(main_set)


# In[25]:


def search_by_substring(sub, array):
    return [x for x in array if sub in x]


# In[26]:


def find_street(streets, street_name, is_street = True):
    """
    Принимает список улиц/районов в искомом районе/Москве streets, а также название улицы или района, введенное пользователем, street_name.
    Если is_street - то улица, иначе район.
    Возвращает название района/улицы, если есть единственный вариант.
    Если вариантов несколько --- возвращает список таких названий.
    Если нет совпадений, то -1.
    """
    if is_street:
        good_str = search_by_substring(sub=street_name.upper(), array=streets)
        if len(good_str) == 0:
            good_str = search_by_substring(sub=district_str_form(street_name),array=streets)
        if len(good_str) == 0:
            good_str = search_by_substring(sub=street_name,array=streets)
    else:
        good_str = search_by_substring(sub=district_str_form(street_name),array=streets)
    if len(good_str) == 1:
        return good_str[0]
    elif len(good_str) > 1:
        return good_str
    
    #на данный момент ничего не найдено
    lists_of_words = []
    street_words = street_name.split()
    
    if len(street_words) == 1:
        if '-' in street_words[0] and street_words[0] != '-':
            street_words = street_words[0].split('-')
            if len(street_words) < 4:
                for word in street_words:
                    rec_result = find_street(streets=streets, street_name=word, is_street = is_street)
                    if type(rec_result) == str:
                        lists_of_words.append([rec_result])
                        continue
                    if rec_result != -1:
                        lists_of_words.append(rec_result)
                if len(lists_of_words) > 0:
                    inter_list = intersect_list(lists_of_words)
                    if len(inter_list) == 1:
                        return inter_list[0]
                    return inter_list
        return -1
    
    if len(street_words) > 5:
        return -1
    
    for word in street_words:
        rec_result = find_street(streets=streets, street_name=word, is_street = is_street)
        if type(rec_result) == str:
                    lists_of_words.append([rec_result])
                    continue
        if rec_result != -1:
            lists_of_words.append(rec_result)
    if len(lists_of_words) == 0:
        return -1
    
    inter_list = intersect_list(lists_of_words)
    if len(inter_list) == 1:
        return inter_list[0]
    return inter_list


# In[27]:


def get_number(str_num):
    """
    Функция принимает на вход текстовую строку и возвращает натуральное число, записанное в начале этой строки
    (т.е. пока не встретится символ, отличный от цифры).
    Если первый символ не является числом, то возвращает None.
    """
    numb = ''
    for n in str_num:
        if n not in DIGITS:
            break
        numb += n
    if len(numb) == 0:
        return None
    return int(numb)


# In[28]:


def get_the_house(hn, houses):
    """
    Принимает номер дома hn(int), а также список всех домов на улице в текстовом формате (как они записаны на сайте ЦИК) houses.
    Возвращает строку из houses, если совпадение по номеру дома единственно,
    список строк из houses, если совпадение не единственно,
    и -1, если совпадений нет.
    """
    houses_on_the_street = [get_number(x) for x in houses]
    if hn in houses_on_the_street:
        indices = [i for i, num in enumerate(houses_on_the_street) if num == hn]
        if len(indices) == 1:
            return houses[indices[0]]
        return [houses[x] for x in indices]
    return -1


# In[29]:


def get_uik_num(text):
    """
    Принимает на вход основной текст веб-страницы с информацией об УИК с сайта ЦИК для найденного адреса text.
    Возвращает номер УИК для наденного адреса(int), либо -1, если данных на сайте ЦИК нет. 
    """
    n_ind = text.find('№')
    if n_ind == -1:
        return -1
    return get_number(text[n_ind+1:])


# In[30]:


def digging(browser, cur_intid):
    """
    Принимает Chrome WebDriver, отрытый на сайте ЦИКа browser, а также intid последнего найденного элемента адреса cur_intid.
    Возвращает основной текст веб-страницы с информацией об УИК с сайта ЦИК для найденного адреса по его последнему элементу.
    """
    ### FROM https://stackoverflow.com/questions/32874539/using-a-variable-in-xpath-in-python-selenium
    browser.find_element_by_xpath("//a[@intid ='" + cur_intid + "']").click()
    ###END FROM
    sleep(2)
    while browser.current_url == CIK_URL:
        source =  BeautifulSoup(browser.page_source)
        cur_intid = source.find(intid = cur_intid).next_sibling.li.a.get('intid')
        browser.find_element_by_xpath("//a[@intid ='" + cur_intid + "']").click()
        sleep(2)
    source =  BeautifulSoup(browser.page_source)
    return source.body.text


# # Работа с API портала открытых данных Москвы 

# In[31]:


import requests


# In[32]:


with open('MOS_KEY.txt') as f:
    MOS_KEY = f.read().split()[0]
MOS_URL = 'https://apidata.mos.ru/v1/datasets/961/rows/'


# In[33]:


def get_uik_address(uik):
    """
    Принимает на вход номер УИК(int).
    Возвращает текстовую строку 'Адрес УИК №uik: ' + адрес УИК из портала открытых данных Москвы,
    либо -1, если такого УИК в базе нет.
    """
    mos_param = {'api_key': MOS_KEY, '$filter': 'Cells/PollStationNumbereq'+str(uik)}
    r_mos_data = requests.get(MOS_URL,params=mos_param)
    mos_data = r_mos_data.json()
    if len(mos_data) == 0:
        return -1
    uik_address = mos_data[0]['Cells']['PollPlaceAddress']
    return 'Адрес УИКа №' + str(uik) + ': ' + uik_address


# # Функции работы с телеграм-ботом

# In[34]:


import telegram
from telegram.error import TimedOut


# In[35]:


with open('BOT_TOKEN.txt') as f:
    BOT_TOKEN = f.read().split()[0]
BOT_URL = 'https://api.telegram.org/bot'+BOT_TOKEN+'/'
GENERAL_COMMANDS = {'help','start','uik','address'}
MOS = "Город Москва"
MAX_BUTTONS = 7
LEVELS = ['district','street', 'street2']
BOT = telegram.Bot(token=BOT_TOKEN)


# In[36]:


RESPONSES = {
    'start': ["Здравствуйте! Вас приветствует телеграм-бот, который предоставит Вам официальную статистику по голосованию в Москве"+
              " на выборах Президента в 2012 и 2018 годах, выборах в Гос. Думу в 2011 и 2016 годах, а также выборах Мэра в 2013 году.",
              "Пожалуйста, следуйте инструкциям бота и не отправляйте больше одного сообщения за раз, не дождавшись ответа от бота"+
              " (бот все равно проигнорирует все сообщения, кроме самого первого)."+
              "\nОтнеситесь с пониманием к тому, что боту может потребоваться некоторое время для сбора и визуализации информации.",
              "На данный момент Вам доступны две опции: введите /uik, если Вы знаете номер УИКа в Москве и желаете получить статистику по нему,"+
              " либо же введите /address, чтобы попытаться получить номер УИКа из базы данных ЦИКа по адресу в Москве."+
    "\nВы всегда можете вернуться в начало с помощью команды /start, а также получить список доступных команд, введя /help. Удачи!"],
    'help': "Для того, чтобы начать все сначала, используйте команду /start."+
    "\nДля того, чтобы получить статистику по конкретному УИКу в Москве по его номеру, используйте команду /uik."+
    "\nДля того, чтобы попытаться получить номер УИКа из базы данных ЦИКа по адресу в Москве, используйте команду /address.",
    'uik': {'start': "Пожалуйста, введите номер интересующего Вас УИКа в Москве для получения статистики по нему.",
            'wrong': "Вы ввели строку, содержащую что-то кроме цифр в качестве номера УИКа. Пожалуйста, используйте только цифры.",
           'no': "К сожалению, УИКа с таким номером не существует. Пожалуйста, введите другой номер УИКа, либо используйте команду /address"+
           " для поиска УИКа по адресу в Москве."},
    'wrong_command': "Введена несуществующая команда. Пожалуйста, попробуйте еще раз.",
    'not_text': "К сожалению, бот способен обрабатывать только текстовые сообщения.",
    LEVELS[0]: {'start': "Пожалуйста, введите район Москвы интересующего Вас дома.",
                 'ok': "Отлично! Теперь введите улицу или населенный пункт, в кототором находится интересующий Вас дом.",
                 'no': "К сожалению, мы не смогли определить интересующий Вас район. Пожалуйста, попробуйте еще раз, либо используйте команду /list, чтобы получить список всех районов Москвы следующим сообщением.",
                 'options': "Мы не смогли однозначно идентифицировать интересующий Вас район, но Вы можете выбрать один из предложенных вариантов, либо вновь написать название района самостоятельно."
                },
    LEVELS[1]: {'start': "Пожалуйста, введите название улицы или населенного пункта интересующего Вас дома.",
                 'ok': "Отлично! Теперь введите номер дома интересующего Вас дома, либо название улицы, на котором он находится (если, например, до этого был указан населенный пункт).",
                 'no': "К сожалению, мы не смогли определить интересующую Вас улицу или населенный пункт. Пожалуйста, попробуйте еще раз, либо используйте команду /list, чтобы получить список всех улиц/населенных пунктов в выбранном районе следующим сообщением.",
                 'options': "Мы не смогли однозначно идентифицировать интересующую Вас улицу/населенный пункт, но Вы можете выбрать один из предложенных вариантов, либо вновь написать название улицы/населенного пункта самостоятельно.",
                'all_options': "Мы не смогли найти или однозначно идентифицировать интересующую Вас улицу/населенный пункт. Пожалуйста, выберите один из вариантов из предложенного списка, т.к. других улиц/населенных пунктов в этом районе в базе ЦИК нет."
                },
    LEVELS[2]: {'start': "Пожалуйста, введите номер интересующего Вас дома, либо название улицы, на котором он находится (если, например, до этого был указан населенный пункт).",
                'ok': "Отлично! Теперь введите номер интересующего Вас дома.",
                'no': "К сожалению, мы не смогли определить интересующий Вас дом или улицу, либо его нет в базе данных ЦИК. Пожалуйста, попробуйте еще раз, либо используйте команду /list, чтобы получить список всех домов/улиц на выбранной улице/населенном пункте следующим сообщением.",
                'options': "Мы не смогли однозначно идентифицировать интересующий Вас дом/улицу, но Вы можете выбрать один из предложенных вариантов, либо вновь написать номер дома/название улицы самостоятельно.",
                'all_options': "Мы не смогли найти или однозначно идентифицировать интересующий Вас дом/улицу. Пожалуйста, выберите один из вариантов из предложенного списка, т.к. других домов/улиц на данной улице/населенном пункте в базе ЦИК нет."
                },
    'cik_no_info': "К сожалению, для данного адреса и соседних домов на данной улице информация об УИКе на сайте ЦИК отсутствует. Нажмите /address для нового запроса, либо /start, для того, чтобы вернуться в начало.",
    'try_neighbour': "К сожалению, на сайте ЦИК не оказалось данных по введенному Вами адресу, однако в данный момент мы пытаемся проверить соседние дома на указанной Вами улице."+
    "\nПожалуйста, подождите, это может занять некоторое время.",
    'pic': "Если Вы хотите получить допольнительные графики, нажмите:"+
    "\n/1, если хотите получить динамику явки на выбранном УИКе,"+
    "\n/2: если хотите получить динамику доли голосов за власть и оппозицию на выбранном УИКе,"+
    "\n/3: если хотите получить распределения явки, голосов за власть и за оппозицию по всем УИКам Москвы,"+
    "\n/end: если хотели бы закончить работу с ботом по данному УИКу.",
    'pic_wrong': "Пожалуйста, используйте одну из предложенных команд для ответа.",
    'end': "Спасибо большое за пользование ботом! Используйте команду /start, если хотите сделать новый запрос."
}


# ### Общие функции телеграм-бота 

# In[37]:


def start_server():
    """
    Запускает сервер, игнорируя все новые сообщения до этого.
    """
    offset = last_update_id(get_updates(url=BOT_URL,timeout=0))
    if offset != None:
        offset +=1
    while True:
        response = get_updates(url=BOT_URL, offset=offset)
        if len(response['result']) > 0:
            offset = response_handler(response)
    return

def get_updates(url, offset=None, timeout = 30):
    """
    Принимает на вход API Telegram с токеном бота url,
    id обновления(int), начиная с которого необходимо получить обновления,
    и время long-pooling timeout(int).
    Возвращает ответ Telegram об обновлениях в формате json.
    """
    params = {'timeout': timeout, 'offset': offset}
    response = requests.get(url + 'getUpdates',params=params)
    return response.json()

def last_update_id(data):
    """
    Принимает на вход JSON ответ с обновлениями от Telegram.
    Возвращает id последнего обновления, либо None, если обновлений нет.
    """
    results = data['result']
    if len(results) > 0:
        total_updates = len(results) - 1
        return results[total_updates]['update_id']
    return None


def get_chat_id(msg):
    """
    Принимает на вход сообщение, присланное пользователем, в формате JSON.
    Возвращает chat_id этого сообщения.
    """
    chat_id = msg['message']['chat']['id']
    return chat_id

def response_handler(response):
    """
    Принимает на вход JSON ответ с обновлениями от Telegram.
    Выделяет из списка обновлений первое сообщение для каждого чата, после чего обрабатывает эти сообщения поочередно.
    Возвращает id обновления поседнего сообщения + 1.
    """
    ids = []
    first_msgs = []
    for msg in response['result']:
        try:
            chat_id = get_chat_id(msg)
        except KeyError:
            continue
        if chat_id in ids:
            continue
        ids.append(chat_id)
        first_msgs.append(msg)
    for msg in first_msgs:
        respond_to_msg(msg)
    return last_update_id(response) + 1

def respond_to_msg(msg):
    """
    Принимает сообщение от пользователя в формате JSON.
    Обрабатывает это сообщение в зависимости от содержания и статуса пользователя.
    Ничего не возвращает.
    """
    chat_id = get_chat_id(msg)
    try:
        record = DB[chat_id]
    except KeyError:
        #новый пользователь
        DB[chat_id] = {'state': 'start'}
        USERS[chat_id] = msg['message']['chat']
        welcome(chat_id)
        return
    
    bot_command = False
    #Получаем текст сообщения и определяем, команда или нет.
    try:
        if msg['message']['entities'][0]['type'] == "bot_command":
            msg_text = msg['message']['text'][1:]
            bot_command = True
    except KeyError:
        try:
            msg_text = msg['message']['text']
        except KeyError:
            BOT.send_message(chat_id=chat_id, text=RESPONSES['not_text'])
            return
    
    #Ответ на общие команды.
    if bot_command and msg_text in GENERAL_COMMANDS:
        command_handler(text = msg_text, chat_id = chat_id)
        return
    
    #Ответ на статус 'pic'
    if record['state'] == 'pic':
        if not bot_command or msg_text not in {'1', '2','3',"end"}:
            BOT.send_message(chat_id=chat_id, text=RESPONSES['pic_wrong'], 
                  reply_markup=telegram.ReplyKeyboardMarkup(keyboard = [['/1'],['/2'],['/3'],['/end']]))
            return
        if msg_text in {'1', '2','3'}:
            DB[chat_id]['current_pic'] = msg_text
            get_uik_stats(uik = record['uik'], chat_id = chat_id)
            return
        delete_files(uik=record['uik'])
        BOT.send_message(chat_id=chat_id, text=RESPONSES['end'], reply_markup=telegram.ReplyKeyboardRemove())
        DB[chat_id] = {'state': 'start'}
        return
    
    
    #Ответ на статус 'uik'
    if record['state'] == 'uik':
        uik_state(chat_id = chat_id, text = msg_text)
        return
    
    #Ответ на статус 'address'
    if record['state'] == 'address':
        if bot_command:
            if msg_text == 'list' and 'cur_list' in record:
                BOT.send_message(chat_id=chat_id, text=', '.join(record['cur_list']), reply_markup=telegram.ReplyKeyboardRemove())
                sleep(1)
                BOT.send_message(chat_id=chat_id, text=RESPONSES[cur_level(chat_id)]['start'])
                return
            BOT.send_message(chat_id=chat_id, text=RESPONSES['wrong_command'],reply_markup=telegram.ReplyKeyboardRemove())
            return
        address_handler(chat_id = chat_id, text = msg_text)
        return
    
    #Введена несуществующая команда.
    if bot_command:
        BOT.send_message(chat_id=chat_id, text=RESPONSES['wrong_command'],reply_markup=telegram.ReplyKeyboardRemove())
        return
    #Введен текст без указания режима.
    BOT.send_message(chat_id=chat_id, text=RESPONSES['start'][2],reply_markup=telegram.ReplyKeyboardRemove())
    return

def welcome(chat_id):
    """
    Принимает chat_id пользователя. Отправляет ему стартовые сообщения. Ничего не возвращает.
    """
    for text in RESPONSES['start']:
        BOT.send_message(chat_id=chat_id, text=text, reply_markup=telegram.ReplyKeyboardRemove())
        sleep(2)
    return

def command_handler(text, chat_id):
    """
    Получает на вход текст одной из основных команд без / text и chat_id пользователя.
    Отвечает шаблонной фразой на каждую из команд, в некоторых случаях меняет статус пользователя.
    Ничего не возвращает.
    """
    if text == 'help':
        BOT.send_message(chat_id=chat_id, text=RESPONSES['help'], reply_markup=telegram.ReplyKeyboardRemove())
        return
    if text == 'start':
        DB[chat_id] = {'state': 'start'}
        welcome(chat_id)
        return
    if text == 'uik':
        DB[chat_id] = {'state': 'uik'}
        BOT.send_message(chat_id=chat_id, text=RESPONSES['uik']['start'],reply_markup=telegram.ReplyKeyboardRemove())
        return
    if text == 'address':
        DB[chat_id] = {'state': 'address'}
        BOT.send_message(chat_id=chat_id, text=RESPONSES[LEVELS[0]]['start'],reply_markup=telegram.ReplyKeyboardRemove())
        return
    BOT.send_message(chat_id=chat_id, text="Я НЕ ДОЛЖЕН БЫЛ ЗДЕСЬ ОКАЗАТЬСЯ, ОСНОВНЫЕ КОМАНДЫ ЗАКОНЧИЛИСЬ",reply_markup=telegram.ReplyKeyboardRemove())
    return


# ### Функции телеграм-бота, связанные с обработкой адреса

# In[38]:


def address_handler(chat_id, text):
    """
    Принимает chat_id и text сообщения. Функция обрабатывает text в состоянии пользователя 'address', в зависимости от того,
    на каком этапе обработки адреса находится пользователь.
    Ничего не возвращает.
    """
    browser = webdriver.Chrome() #не забыть дать ссылку на установку
    browser.get(CIK_URL)
    sleep(2)
    browser.find_element_by_link_text(MOS).click()
    sleep(2)
    #проверяем, был ли ранее указан район, если да, то улица/населенный пункт. Если нет, то пытаемся выяснить. 
    for level in LEVELS[0:2]:
        if level not in DB[chat_id]:
            disstr_handler(browser=browser, level = level, text=text, chat_id=chat_id)
            browser.quit()
            return
        browser.find_element_by_link_text(DB[chat_id][level]).click()
        sleep(2)
    #на данный момент у нас имеются как район, так и улица/населенный пункт
    if LEVELS[2] not in DB[chat_id]:
        #теперь мы пытаемся выяснить, вводит ли человек номер дома, либо же улицу
        if get_number(text) == None or len(text) > 7:
            #скорее всего, улица
            disstr_handler(browser=browser, level=LEVELS[2], text=text, chat_id=chat_id)
            browser.quit()
            return
        #скорее всего, номер дома
        number_handler(browser = browser, last_level = 1, text = text, chat_id = chat_id)
        return
    browser.find_element_by_link_text(DB[chat_id][LEVELS[2]]).click()
    sleep(2)
    #если мы тут, то было введен район, населенный пункт, и улица, так что теперь обрабатываем текст как номер дома.
    number_handler(browser=browser, last_level=2, text=text, chat_id=chat_id)    
    return

def disstr_handler(browser, level, text, chat_id):
    """
    Функция принимает открытый на сайте ЦИК Chrome WebDriver browser,
    уровень адреса, который мы пытаемся идентифицировать, level,
    текст сообщения пользователя text и chat_id.
    Функция с помощью текстовых обработчиков пытается сопоставить введенный пользователем текст с текстом на сайте ЦИК.
    Ничего не возвращает.
    """
    if 'cur_list' not in DB[chat_id]:
        source = BeautifulSoup(browser.page_source)
        if level == LEVELS[0]:
            DB[chat_id]['cur_list'] = [li.text for li in source.find_all(string = MOS)[0].parent.next_sibling.find_all('li')]
        else:
            for i in range(2):
                if level == LEVELS[i+1]:
                    DB[chat_id]['cur_list'] = [li.text for li in source.find_all(string = DB[chat_id][LEVELS[i]])[0].parent.next_sibling.find_all('li')]
                    break
    if level == LEVELS[0]:
        info = find_street(streets = DB[chat_id]['cur_list'], street_name = text, is_street = False)
    else:
        info = find_street(streets = DB[chat_id]['cur_list'], street_name = text)   
    if type(info) == str:
        #однознчно идентифицировали, поэтому добавляем в базу данных
        DB[chat_id][level] = info
        BOT.send_message(chat_id=chat_id, text=RESPONSES[level]['ok'], reply_markup=telegram.ReplyKeyboardRemove())
        if 'cur_list' in DB[chat_id]:
            del DB[chat_id]['cur_list']
        return
    
    if info == -1:
        #нет схожих результатов, если кол-во вариантов мало, то клавиатура, иначе отказ
        if len(DB[chat_id]['cur_list']) <= MAX_BUTTONS:
            info = [[x] for x in DB[chat_id]['cur_list']]
            BOT.send_message(chat_id=chat_id, text=RESPONSES[level]['all_options'], 
                  reply_markup=telegram.ReplyKeyboardMarkup(keyboard = info, one_time_keyboard = True))
            return
        BOT.send_message(chat_id=chat_id, text=RESPONSES[level]['no'], reply_markup=telegram.ReplyKeyboardRemove())
        return
    #если вариантов слишком много, то нет клавиатуры
    if len(info) > MAX_BUTTONS:
        BOT.send_message(chat_id=chat_id, text=RESPONSES[level]['no'], reply_markup=telegram.ReplyKeyboardRemove())
        return
    #отправляем пользователя потенциальные варианты в виде клавиатуры
    info = [[x] for x in info]
    BOT.send_message(chat_id=chat_id, text=RESPONSES[level]['options'], 
                  reply_markup=telegram.ReplyKeyboardMarkup(keyboard = info, one_time_keyboard = True))
    return

def number_handler(browser, last_level, text, chat_id):
    """
    Функция принимает открытый на сайте ЦИК Chrome WebDriver browser,
    уровень адреса, который был идентифицирован последним, last_level,
    текст сообщения пользователя text и chat_id.
    Функция с помощью текстовых обработчиков пытается сопоставить введенный пользователем текст с номером дома на сайте ЦИК.
    Ничего не возвращает.
    """
    #Получаем список всех домов на улице, если до этого его не было.
    if 'cur_list' not in DB[chat_id]:
        source =  BeautifulSoup(browser.page_source)
        DB[chat_id]['cur_list'] = [li.text for li in source.find_all(string = DB[chat_id][LEVELS[last_level]])[0].parent.next_sibling.find_all('li')]
    houses = DB[chat_id]['cur_list']
    #если есть совпадение, то пытаемся добыть номер УИК
    if text in houses:
        uik_address(browser=browser, last_level=last_level, text=text, chat_id=chat_id)
        return
    hn = get_number(text)
    #текстовых совпадений нет, пытаемся по цифрам
    if hn != None:
        house = get_the_house(hn=hn, houses=houses)
        #если однозначное совпадение, то пытаемся добыть номер УИК
        if type(house) == str:
            uik_address(browser=browser, last_level=last_level, text=house, chat_id=chat_id)
            return
        #если много домов с таким номером, отправляем пользователю список
        if type(house) == list:
            browser.quit()
            info = [[x] for x in house]
            BOT.send_message(chat_id=chat_id, text=RESPONSES[LEVELS[-1]]['options'], 
                  reply_markup=telegram.ReplyKeyboardMarkup(keyboard = info, one_time_keyboard = True))
            return
    browser.quit()
    #если домов на улице не очень много, то отправляем пользователю список всех домов
    if len(houses) <= MAX_BUTTONS:
        info = [[x] for x in houses]
        BOT.send_message(chat_id=chat_id, text=RESPONSES[LEVELS[-1]]['all_options'], 
                  reply_markup=telegram.ReplyKeyboardMarkup(keyboard = info, one_time_keyboard = True))
        return
    
    BOT.send_message(chat_id=chat_id, text=RESPONSES[LEVELS[-1]]['no'], 
                  reply_markup=telegram.ReplyKeyboardRemove())
    return
        
    
def uik_address(browser, last_level, text, chat_id):
    """
    Функция принимает открытый на сайте ЦИК Chrome WebDriver browser,
    уровень адреса, который был идентифицирован последним, last_level,
    найденный номер дома text и chat_id.
    Функция пытается достать номер УИК для найденного адреса.
    Ничего не возвращает.
    """
    source =  BeautifulSoup(browser.page_source)
    houses = DB[chat_id]['cur_list']
    uik_num = get_uik_num(digging(browser=browser,
                                      cur_intid= source.find_all(string = DB[chat_id][LEVELS[last_level]])[0].parent.next_sibling.find_all('li')[houses.index(text)].a.get('intid')))
    if uik_num == -1:
        #информации на сайте ЦИК нет
        if len(houses) == 1:
            browser.quit()
            #дом на улице единственный, ничего с этим не поделать, придется начинать заново.
            BOT.send_message(chat_id=chat_id, text=RESPONSES['cik_no_info'], reply_markup=telegram.ReplyKeyboardRemove())
            DB[chat_id] = {'state': 'start'}
            return
        #домов несколько, пытаемся пройтись по всем соседям до первого найденного УИКа
        BOT.send_message(chat_id=chat_id, text=RESPONSES['try_neighbour'], reply_markup=telegram.ReplyKeyboardRemove())
        address = [MOS]
        for i in range(last_level+1):
            address.append(DB[chat_id][LEVELS[i]])
        uik_num = get_neighbours_uiks(browser=browser, address=address, houses=houses, index=houses.index(text))
        #если у ЦИК нет информации так же ни на одного из соседей, то опять ничего с этим не сделать
        if uik_num == -1:
            browser.quit()
            BOT.send_message(chat_id=chat_id, text=RESPONSES['cik_no_info'], reply_markup=telegram.ReplyKeyboardRemove())
            DB[chat_id] = {'state': 'start'}
            return
    
    browser.quit()
    #если же уик нашелся, пытаемся получить статистику    
    if get_uik_stats(uik=uik_num, chat_id=chat_id) == -1:
        bot.send_message(chat_id=chat_id, text="Каким-то образом УИКа с сайта ЦИКа не оказалось в базе данных. УИК №"+str(uik_num), 
                reply_markup=telegram.ReplyKeyboardRemove())
    return
    
def get_neighbours_uiks(browser, address, houses, index):
    """
    Функция принимает открытый на сайте ЦИК Chrome WebDriver browser,
    последовательный список уровней адреса на сайте, address,
    список домов на улице, houses,
    и индекс нашего дома в этом списке.
    Функция пытается достать номер УИК для соседних домов, двигаясь влево (в сторону уменьшения адреса).
    Если номер УИК есть для одного из соседей, то он и возвращается, иначе -1.
    """
    for i in range(1,len(houses)):
        browser.back()
        sleep(2)
        for addr in address:
            browser.find_element_by_link_text(addr).click()
            sleep(2)
        source =  BeautifulSoup(browser.page_source)
        uik_num = get_uik_num(digging(browser=browser,
                                      cur_intid = source.find_all(string=address[-1])[0].parent.next_sibling.find_all('li')[index-i].a.get('intid')))
        if uik_num != -1:
            return uik_num
    return -1

def cur_level(chat_id):
    """
    Принимает chat_id.
    Возвращает уровень адреса, который мы пытаемся идентефицировать.
    Если все идентифицированы, то возвращает последний возможный уровень.
    """
    for level in LEVELS:
        if level not in DB[chat_id]:
            return level
    return LEVELS[2]


# ### Функции телеграм-бота, связанные с обработкой УИК

# In[39]:


def uik_state(chat_id, text):
    """
    Принимает chat_id и текст сообщения пользователя в состоянии 'uik', text.
    Если введен корректный номер УИК отправляет графики,
    если такого уика нет, либо введен не номер, то делает соответствующее заявление.
    Ничего не возвращает
    """
    try:
        stats = get_uik_stats(uik=int(text), chat_id=chat_id)
        if stats == -1:
            BOT.send_message(chat_id=chat_id, text=RESPONSES['uik']['no'], reply_markup=telegram.ReplyKeyboardRemove())
            return
        return
    except ValueError:
        BOT.send_message(chat_id=chat_id, text=RESPONSES['uik']['wrong'], reply_markup=telegram.ReplyKeyboardRemove())
        return

def get_uik_stats(uik, chat_id):
    """
    Принимает номер УИКа(int) и chat_id.
    Отправляет адрес УИК, если он есть в базе открытых даннных Москвы.
    Если УИК есть в нашей базе данных, то отправляет круговую диаграмму результатов голосования
    и предлагает прислать другую статистику.
    Если же диаграмма была отправлена ранее, то отправляет пользователю график, который он выбрал в своем сообщении,
    предлагая вновь прислать все виды графиков.
    Возвращает -1, если УИКа нет в базе данных.
    """
    #пользователь еще не получил круговую диаграмму
    if DB[chat_id]['state'] != 'pic':
        DB[chat_id]['uik'] = uik
        adr_uik = get_uik_address(uik)
        if adr_uik != -1:
            BOT.send_message(chat_id=chat_id, text=adr_uik, reply_markup=telegram.ReplyKeyboardRemove())
        if pies_for_uik(uik) == -1:
            return -1
        send_picture(uik=uik,chat_id=chat_id,filename=FILENAMES[0],distr=False)
        DB[chat_id]['state'] = 'pic'
        sleep(1)
        BOT.send_message(chat_id=chat_id, text=RESPONSES['pic'], 
                  reply_markup=telegram.ReplyKeyboardMarkup(keyboard = [['/1'],['/2'],['/3'],['/end']]))
        return
    #пользователь запросил динамику явки
    if DB[chat_id]['current_pic'] == '1':
        if not os.path.exists(FILENAMES[1]+str(uik)+'.png'):
            if dynamics(collect_dyn_data(uik)) == -1:
                BOT.send_message(chat_id=chat_id, text='Каким-то образом УИК исчез из базы данных на этапе динамики.', reply_markup=telegram.ReplyKeyboardRemove())
                return -1
        send_picture(uik=uik,chat_id=chat_id,filename=FILENAMES[1], distr=False)
    #пользователь запросил динамику голосования за власть и оппозицию     
    if DB[chat_id]['current_pic'] == '2':
        if not os.path.exists(FILENAMES[2]+str(uik)+'.png'):
            if dynamics(collect_dyn_data(uik), turnout = False) == -1:
                BOT.send_message(chat_id=chat_id, text='Каким-то образом УИК исчез из базы данных на этапе динамики.', reply_markup=telegram.ReplyKeyboardRemove())
                return -1
        send_picture(uik=uik,chat_id=chat_id,filename=FILENAMES[2],distr=False)
    #пользователь запросил распределения явки и голосов за власть и оппозицию по УИКам
    if DB[chat_id]['current_pic'] == '3':
        if not os.path.exists('distributions.png'):
            get_distributions() 
        send_picture(uik=uik,chat_id=chat_id)
    #вновь предлагаем выбрать график
    sleep(1)
    BOT.send_message(chat_id=chat_id, text=RESPONSES['pic'],
                     reply_markup=telegram.ReplyKeyboardMarkup(keyboard = [['/1'],['/2'],['/3'],['/end']]))
    return

def send_picture(uik, chat_id,filename='distributions.png', distr = True):
    """
    Принимает номер уика(int), chat_id, директорию, в которой находится файлы для uik (downloads по умолчанию),
    а также имя отправляемого файла (если это не distributions, что по умолчанию), а также флаг distr (по умолчанию True),
    если отправляется распределение или нет.
    Отправляет соответствующий файл пользователю.
    """
    if distr:
        pic_name = filename
    else:
        pic_name = filename+str(uik)+'.png'
    if os.path.exists(pic_name):
        pic = open(pic_name, 'rb')
        try:
            if filename == FILENAMES[0]:
                BOT.send_document(chat_id=chat_id, document=pic)
            else:
                BOT.send_photo(chat_id=chat_id, photo=pic)
        except TimedOut:
            pic.close()
            return
        pic.close()
    return


# # Тело

# In[40]:


USERS = dict()


# In[41]:


DB = dict()


# In[42]:


def main():
    try:
        start_server()
    except TimedOut:
        main()
    except KeyboardInterrupt:
        print("Сервер остановлен пользователем")


# In[43]:


main()

