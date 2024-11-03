# Parkinsons Telemonitoring

**Report per l’Esame di Fondamenti di Machine Learning**

**Davide Paltrinieri**  
Matricola: 165005  
Ingegneria Informatica  
Email: 296723@studenti.unimore.it  
Data: June 2024  

## Abstract

Il seguente documento descrive il percorso e le scelte progettuali adottate per lo creazione del progetto d’esame di Fondamenti di Machine Learning, relativamente all’appello del 17/07/2024.

## 1 Introduzione al problema

Il dataset Parkinsons Telemonitoring raccoglie dati biomedicali riguardanti gli stadi iniziali della malattia del Parkinsons. In particolare, il dataset contiene informazioni relative a 42 diversi pazienti, e per ognuno di essi possiede circa 200 samples, per un totale di 5875. Il principale obiettivo è quello di predire le variabili target motor UPDRS e total UPDRS, le quali fanno riferimento alla scala UPDRS (Unified Parkinson’s Disease Rating Scale), che è una scala comunemente utilizzata per monitorare la progressione della malattia di Parkinson.

## 2 Struttura del progetto

Il progetto è strutturato nel seguente modo:

- `train.py`
- Cartella `pt_dataset`, contenente il dataset (sia in formato `.csv` sia in formato `.data` e `.names`)
- `data.log`, file di logging contenente tutte le informazioni sui modelli, sull’esecuzione del programma e sui risultati ottenuti passo dopo passo. Ad ogni esecuzione viene pulito e riscritto con le nuove informazioni, e se mancante viene creato.
- Cartella `graphs`, contenente i grafici generati al termine dell’esecuzione del codice (riportati nel paragrafo 5).

Per l’avvio sarà sufficiente mandare in esecuzione `train.py`, anche aggiungendo come parametro il dataset (in caso non venisse inserito il percorso `pt_dataset\parkinsons_updrs.data` verrà considerato come valore di default).

## 3 Analisi e gestione dei dati

### 3.1 EDA

Come descritto in precedenza il dataset contiene 5875 samples, ognuno dei quali è composto da 20 features principalmente intere (int64) e reali (float64) e una variabile categorica. Di seguito vengono descritte in breve:

- `subject number`, un numero compreso tra 1 e 42 che indica il numero del soggetto identificandolo (int64)
- `subject age`
- `subject gender`, unica feature categorica
- `test time`, tempo di partecipazione allo studio
- 16 biomedical measures (tutte float64)

Una nota importante è che il dataset risulta completo, nessun sample ha valori mancanti. Il dataset è stato suddiviso in Training (70%) e Testing (30%).

### 3.2 Scalamento dei dati

Analizzando le informazioni contenute nel dataset risulta chiaro che le features hanno scale di valori compresi in range differenti tra loro, per questo motivo ho preferito procedere con un processo di scalamento dei dati, ovvero la **Standardizzazione**.

### 3.3 Feature selection

Osservando il numero di features in relazione al numero di samples presenti nel dataset non sembrava essere strettamente necessario un processo di feature selection. Ciononostante, ho preferito procedere con tale operazione, ottenendo un miglioramento nei risultati finali. In particolare, dato che testiamo modelli differenti, ho deciso di adottare una metodologia del tipo Wrapper Method in modo da adattare la feature selection al modello specifico considerato. Inoltre, ho ritenuto opportuno eliminare dal dataset la feature `subject number` perché ritenuta particolarmente inefficace al fine delle predizioni. Quindi, ogni modello descritto al paragrafo 4.2, è stato provato e valutato sia con che senza feature selection. I risultati ottenuti sono stati riportati nel paragrafo 4.3.

## 4 Cross Validation e prova dei modelli

In questa sezione verranno indicati i modelli testati e le metriche utilizzate, con i relativi risultati generati nella fase di Cross Validation.

### 4.1 Metriche

Per la valutazione dei modelli sono state adottate le seguenti metriche:

- `neg mean absolute error`
- `neg mean squared error`
- `neg root mean squared error`
- `r2`

### 4.2 Modelli e iperparametri

I modelli provati sono molteplici, ognuno dei quali è stato testato due volte, rispettivamente con e senza feature selection. In particolare, per ogni modello, sono state generate due pipeline: la prima contenente il regressore e un Sequential Feature Selector, la seconda contenente solo il modello. Successivamente, su ciascuna pipeline è stata avviata una grid search con lo scopo di trovare la combinazione migliore di iperparametri che permettesse di generalizzare al meglio i dati. Di seguito vengono mostrati i modelli adottati con i parametri provati:

- **Linear Regression**
- **K-Nearest Neighbors**  
  - K = [3, 5, 7, 9]
- **Support Vector Regressor**  
  - C = [1, 10]  
  - gamma = [0.001, 0.01]
- **Decision Tree**  
  - max depth = [10, 20, None]  
  - min samples split = [2, 5]
- **Random Forest**  
  - numero estimators = [50, 100, 200]  
  - min samples split = [2, 5, 10]

### 4.3 Risultati

#### 4.3.1 motor UPDRS

In seguito ai processi descritti al paragrafo precedente (4.2) è stato eseguito un processo di cross validation per la scelta del modello ottimale.  
Nota: le tabelle seguenti riguardano la variabile target motor UPDRS.

**Table 1: motor UPDRS con feature selection**

| Modello               | neg MAE   | neg MSE   | neg RMSE  | R2      |
|-----------------------|-----------|-----------|-----------|---------|
| LinearRegression      | -6.4127   | -57.3823  | -7.5722   | 0.1370  |
| KNeighborsRegressor   | -3.1961   | -21.9084  | -4.6771   | 0.6698  |
| SVR                   | -4.8802   | -39.2567  | -6.2644   | 0.4090  |
| DecisionTreeRegressor | -0.5196   | -3.4268   | -1.8450   | 0.9485  |
| RandomForestRegressor | -0.6011   | -1.9606   | -1.3972   | 0.9705  |

**Table 2: motor UPDRS senza feature selection**

| Modello               | neg MAE   | neg MSE   | neg RMSE  | R2      |
|-----------------------|-----------|-----------|-----------|---------|
| LinearRegression      | -6.3639   | -56.8181  | -7.5341   | 0.1456  |
| KNeighborsRegressor   | -3.1725   | -24.2967  | -4.9276   | 0.6343  |
| SVR                   | -5.3555   | -46.3815  | -6.8088   | 0.3020  |
| DecisionTreeRegressor | -0.7925   | -6.4843   | -2.5272   | 0.9029  |
| RandomForestRegressor | -0.8700   | -2.8121   | -1.6753   | 0.9576  |

#### 4.3.2 total UPDRS

**Table 3: total UPDRS con feature selection**

| Modello               | neg MAE   | neg MSE   | neg RMSE  | R2      |
|-----------------------|-----------|-----------|-----------|---------|
| LinearRegression      | -8.1429   | -97.1317  | -9.8520   | 0.1580  |
| KNeighborsRegressor   | -4.1199   | -37.5284  | -6.1237   | 0.6737  |
| SVR                   | -6.5175   | -77.0747  | -8.7762   | 0.3319  |
| DecisionTreeRegressor | -0.6273   | -5.2887   | -2.2959   | 0.9541  |
| RandomForestRegressor | -0.8147   | -3.5844   | -1.8894   | 0.9689  |

**Table 4: total UPDRS senza feature selection**

| Modello               | neg MAE   | neg MSE   | neg RMSE  | R2      |
|-----------------------|-----------|-----------|-----------|---------|
| LinearRegression      | -8.1095   | -96.3776  | -9.8134   | 0.1645  |
| KNeighborsRegressor   | -4.2663   | -44.9509  | -6.7042   | 0.6098  |
| SVR                   | -7.0332   | -87.5793  | -9.3551   | 0.2409  |
| DecisionTreeRegressor | -0.9981   | -10.3635  | -3.1671   | 0.9093  |
| RandomForestRegressor | -1.0690   | -4.5575   | -2.1306   | 0.9604  |

## 5 Conclusioni

In questa sezione andiamo a trarre le conclusioni prendendo in considerazione le informazioni e i dati osservati in precedenza. Prestando attenzione alle metriche riportate nelle tabelle possiamo facilmente notare un notevole miglioramento nei risultati quando si utilizzano modelli come il Decision Tree e il Random Forset. Per una migliore visualizzazione si possono prendere in considerazione i due grafici successivi. Questi rappresentano, rispettivamente per motor UPDRS e total UPDRS, un insieme di punti, dove la coordinata x indica la variabile target predetta sul dataset di testing, mentre la coordinata y indica la variabile target effettiva del dataset di testing. Quindi un modello sarà più efficace quanto più la linea sarà retta e con un’angolazione di 45°.

Comunque, è degno di nota il fatto che per entrambe le variabili target il modello ottimale risultante è il Random Forset con feature selection, oltretutto ottenendo valide performance sul dataset di testing. Di seguito i risultati effettivi sul dataset di testing e le feature selezionate:

**Model motor UPDRS**  
neg MAE -0.493563360181508  
neg MSE -1.306831472688216  
neg RMSE -1.1431672986436483  
R2 0.9799484245181771  

**Table 5: motor UPDRS risultati finali**

**Model total UPDRS**  
neg MAE -0.566947604651162  
neg MSE -1.8880643708320002  
neg RMSE -1.3740685466278602  
R2 0.9832142624649486  

## Grafici e Visualizzazioni

![Grafico Motor UPDRS](images/motor_updrs.png)

![Grafico Total UPDRS](images/total_updrs.png)


**Table 6: total UPDRS risultati finali**

Le feature selezionate sono:

- motor UPDRS: age, sex, test time, Jitter(%), Jitter(Abs), Jitter:PPQ5, NHR, RPDE, DFA
- total UPDRS: age, sex, test time, Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP, DFA

## References

Athanasios Tsanas and Max Little. Parkinsons Telemonitoring. UCI Machine Learning Repository, 2009. DOI: [https://doi.org/10.24432/C5ZS3N](https://doi.org/10.24432/C5ZS3N).
