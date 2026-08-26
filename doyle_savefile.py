"""
Traccia le spirali logaritmiche e i cerchi di Doyle per dati valori interi p e q (0 <= p <= q)
e salva la figura in formato SVG nella cartella del file Python.
@author: Lorenzo Roi
versione 1
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from pathlib import Path

#################################################################################################################

# definisce le equazioni da risolvere. Testo: eq. 2.38
def sistema_equazioni(variabili, p, q):
    a, alpha = variabili
    b = a**(p/q)
    beta = (2*np.pi+p*alpha)/q
    rQuadrato = (1+a**2-2*a*np.cos(alpha))/(1+a)**2
    eq1 = rQuadrato - (1 + b**2 - 2*b*np.cos(beta))/(1 + b)**2
    eq2 = rQuadrato - (a**2+b**2-2*a*b*np.cos(beta-alpha))/(a+b)**2
    return [eq1, eq2]

# risolve numericamente il sistema di equazioni per trovare a, alpha, b, beta e raggio
def doyle(p, q):
    valori_iniziali = [1.235, 0]
    a, alpha = fsolve(sistema_equazioni, valori_iniziali, args=(p, q))
    b = a**(p/q)
    beta = (2*np.pi + p*alpha)/q 
    a_complex = complex(a*np.cos(alpha), a*np.sin(alpha))
    b_complex = complex(b*np.cos(beta),  b*np.sin(beta))
    raggio = np.sqrt((1 + a**2 - 2*a*np.cos(alpha))/(1 + a)**2)
    return a_complex, b_complex, raggio, a, alpha, b, beta

# seleziona i centri dei cerchi e i raggi corrispondenti in base alla distanza dall'origine
def seleziona(estremo_sup_indici, estremo_inf_abs, estremo_sup_abs, a_complex, b_complex, raggio):
    centri_raggi = []
    for k in range(-estremo_sup_indici, estremo_sup_indici):
        for i in range(-estremo_sup_indici, estremo_sup_indici):
            z = (a_complex**i)*(b_complex**k)
            if (abs(z) >= estremo_inf_abs) and (abs(z) <= estremo_sup_abs) and not((i == 0) and (k == 0)):
                centri_raggi.append((z.real, z.imag, abs(raggio*z), k, i))
    if not centri_raggi:
        return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]))
    ascisse, ordinate, raggi, k_effettivi, i_effettivi = np.array(centri_raggi).T
    return ascisse, ordinate, raggi, k_effettivi, i_effettivi

# controlla se l'input è valido e restituisce il valore di default se l'input è vuoto
def input_default(prompt, default, tipo):
    while True:
        valore = input(f'{prompt} ({default}): ').strip()
        if not valore:
            return default
        try:
            return tipo(valore)
        except ValueError:
            print('Valore non valido, riprovare.')

############## INPUT ###################################################################################################

print("\nInserire i valori interi di p e q (0 <= p <= q),\nl''estremo superiore intero degli indici,\nla distanza (real) minima e massima dall''origine e lo zoom (real).\n")
p = input_default('inserire p:', 5, int)
q = input_default('inserire q:', 13, int)
if q <= 0 or p < 0 or p > q:
    raise ValueError('Occorre avere 0 <= p <= q')
estremo_sup_indici = input_default('estremo sup. indici (>q):', 30, int)
if estremo_sup_indici <= q:
    raise ValueError('Conviene avere estremo sup. indici > q')
estremo_inf_abs = input_default('estremo inf. distanza:', 0.1, float)
estremo_sup_abs = input_default('estremo sup. distanza:', 13, float)
indice_spirale = input_default('1 (p), 2 (q), 3 (q-p):', 1, int)
zoom = input_default('zoom:', 6, float)

############# CALCOLO ###################################################################################################

a_complex, b_complex, raggio, a, alpha, b, beta = doyle(p,q)
ascisse, ordinate, raggi, k_effettivi, i_effettivi  = seleziona(estremo_sup_indici, estremo_inf_abs, estremo_sup_abs, a_complex, b_complex, raggio)

############# GRAFICA ###################################################################################################

fig, ax = plt.subplots(figsize = (8,8))
if (indice_spirale == 1):
    # spirale logaritmica 1
    theta = np.linspace(-20*estremo_inf_abs, 10*estremo_sup_abs, 1000)
    for i in range(p):
        x = b**theta*np.cos(beta*theta + (2*np.pi/p)*i)
        y = b**theta*np.sin(beta*theta + (2*np.pi/p)*i)
        plt.plot(x, y, color = 'blue', linewidth = .5, zorder = 1)
elif (indice_spirale == 2):
    # spirale logaritmica 2
    theta = np.linspace(-20*estremo_inf_abs, 10*estremo_sup_abs, 1000)
    for i in range(q):
        x = a**theta*np.cos(alpha*theta + (2*np.pi/q)*i)
        y = a**theta*np.sin(alpha*theta + (2*np.pi/q)*i)
        plt.plot(x, y, color = 'red', linewidth = .5, zorder = 1)
elif (indice_spirale == 3):
    # spirale logaritmica 3
    theta = np.linspace(-20*estremo_inf_abs, 10*estremo_sup_abs, 1000)
    for i in range(q-p):
        x = (a/b)**theta*np.cos((alpha-beta)*theta + (2*np.pi/(q-p))*i)
        y = (a/b)**theta*np.sin((alpha-beta)*theta + (2*np.pi/(q-p))*i)
        plt.plot(x, y, color = 'teal', linewidth = .5, zorder = 1)
# seleziona la colorazione dei cerchi
cmap = plt.get_cmap('hsv')
for x, y, r, kk, ii in zip(ascisse, ordinate, raggi, k_effettivi, i_effettivi):
    if indice_spirale == 1:
        indice, modulo = ii, p
        #in tal caso non c'è la spirale logaritmicha, quindi usiamo q come modulo per colorare le corone di cerchi concentriche
        if p == 0: 
            indice, modulo = ii, q
    elif indice_spirale == 2:
        indice, modulo = kk, q
    else:
        indice, modulo = ii + kk, q - p
        #in tal caso non c'è la  spirale logaritmica, quindi usiamo q come modulo per colorare le corone di cerchi concentriche
        if p == q:
            indice, modulo = ii + kk, q
    colore = cmap(plt.Normalize(0, modulo)(indice % modulo))
    cerchio = plt.Circle((x, y), r,  ec = 'gray', color = colore, fill = True, alpha = .35, linewidth = 1, zorder = 0)
    plt.gca().add_patch(cerchio)
    plt.scatter(x, y, c = 'w', marker = '.', s = 4, zorder = 2)
plt.xlim([-zoom, zoom])
plt.ylim([-zoom, zoom])
plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False) 
ax.set_aspect('equal')
plt.text(.02, -.04, 'p = '+str(p)+', q = '+str(q)+', spirale = '+str(indice_spirale)+', intervallo = ('+str(estremo_inf_abs)+',' +str(estremo_sup_abs)+'), zoom = '+str(zoom), transform = ax.transAxes)

#costruisce il nome del file in cui salvare la figura
nome_file = Path(__file__).parent / f'p{p}_q{q}_spirale{indice_spirale}.svg'
plt.savefig(nome_file, format='svg', dpi=1200, bbox_inches='tight')