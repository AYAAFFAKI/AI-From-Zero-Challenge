import pandas as pd
import numpy  as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kstest, norm,stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Dataset_Fraude_Bancaire_complete.csv')
#Affiche les 5 lignes premiers
df.head()

#Le size de DataFram:
print(f"\t==Le size de DataFram est: {df.shape}")

#Affiches les informations de df:
print("\t==Les informations de DataFram:")
df.info()

print('\t==Les colonnes qui ont des valeur null:')
print(df.isnull().sum())

print("\t==Les statistique de données:")
df.describe(include='all')

#
tagert = "is_fraud"
maxi = df[df[tagert]==0]
mini = df[df[tagert]==1]
#Affiche la distrébution de variable cible
print("=Les observation normale sont: ",maxi.shape[0],"\n=Porsentage: ",maxi.shape[0]/1000*100,"%" )
print("+Les observation fraud sont: ",mini.shape[0],"\n+Porsentage: ",mini.shape[0]/1000 *100,"%")

sns.countplot(x=tagert , data=df, palette=['#5AEB74','#EB4B4B'])


#Les colonnes numériques:
col_num = df.select_dtypes(include=['int64','float64']).copy()

#supperimer le variable cible :
if tagert in col_num.columns:
    col_num.drop(tagert,axis=1,inplace=True)
#voir les statistique :
col_num.describe()


#Les colonnes catégorique
col_cat = df.select_dtypes(include=['object', 'string']).copy()
#Separer les colonne temporelle:
col_temp = col_cat.select_dtypes(include=['datetime64']).copy()
for col in col_cat.columns:
        if 'date' in col.lower() or 'heure' in col.lower():
            col_temp[col] = col_cat[col]
            col_cat.drop(columns=col, inplace=True)

col_cat.drop(columns=['transaction_id','client_id'],inplace=True)

# Dictionnaire global de NaN
nan_dict = {col: df[col].isna().sum() for col in df.columns}
# Proportion des valeurs catégorielles
unique_cat_ratio = {}
for col in col_cat:
    unique_cat_ratio[col] = df[col].value_counts(normalize=True, dropna=False).to_dict()

for col, ratio_dict in unique_cat_ratio.items():
    print(f"{col} : {ratio_dict}")

threshold_nan = 3       # % pour alerter / supprimer
dominant_threshold = 0.1  # pour valeur dominante

# --- Fonction de suppression si peu de NaN ---
def suppression():
    for col in df.columns:
        nb_nan = nan_dict[col]
        perc_nan = nb_nan / df.shape[0] * 100
        if perc_nan < threshold_nan and nb_nan > 0:
            print(f"Suppression NaN <{threshold_nan}% en colonne : {col}")
            df.dropna(subset=[col], inplace=True)
            
# --- Fonction d'imputation ---
def imputationNum():
    # Numérique
    for col in col_num.columns:
        nb_nan = nan_dict[col]
        perc_nan = nb_nan / df.shape[0] * 100
        if perc_nan > threshold_nan:
            col_data = col_num[col]
            stat, p = stats.normaltest(col_data)
            mean = col_data.mean()
            if p > 0.05:
                print(f"Imputation NaN <{threshold_nan}% en colonne : {col} -> moyenne")
                col_num[col].fillna(mean, inplace=True)
            else:
                print(f"Imputation NaN <{threshold_nan}% en colonne : {col} -> median")
                col_num[col].fillna(col_data.median(), inplace=True)
                
def imputationCat():
    # Catégorique
  for col in col_cat.columns:
        nb_nan = nan_dict[col]                     # nombre de NaN dans la colonne
        perc_nan = nb_nan / df.shape[0] * 100     # % de NaN
        if perc_nan > threshold_nan:              # si % de NaN > seuil global
            top_value = max(unique_cat_ratio[col], key=unique_cat_ratio[col].get)  # modalité dominante
            top_ratio = unique_cat_ratio[col][top_value]   # ratio de cette modalité
            if top_ratio > dominant_threshold:
                print(f"Imputation NaN <{threshold_nan}% en colonne : {col} -> valeur freq")
                # La modalité dominante est assez fréquente → on remplace NaN par cette valeur
                col_cat[col].fillna(df[col].mode()[0], inplace=True)
            else:
                print(f"Imputation NaN <{threshold_nan}% en colonne : {col} -> valeur 'Unknown'")
                # La modalité dominante n'est pas suffisamment forte → remplacer par 'Unknown'
                col_cat[col].fillna('Unknown', inplace=True)

def imputationTemp(df):
  nan_dict = {col: df[col].isna().sum() for col in df.columns}
  for col in col_temp:
      nb_nan = nan_dict[col]
      perc_nan = nb_nan / df.shape[0] * 100
      if perc_nan > threshold_nan:
        print(f"Imputation NaN <{threshold_nan}% en colonne : {col} -> interpolate")
        df[col] = df[col].interpolate(method='linear')
        

def choixStdNor():
  results = {}
   # Test de normalité pour chaque colonne
  for col in col_num.columns:
        if col_num[col].dtype == 'int64':  # éviter les colonnes discrètes
            continue
        stat, p = stats.normaltest(col_num[col])
        results[col] = {'statistic': stat, 'p_value': p, 'is_normal': p > 0.05}

    # Application du bon type de normalisation
  for col, result in results.items():
     if col_num[col].dtype == 'int64':  # éviter les colonnes discrètes
            continue
     else:
        if result['is_normal']:
            print(f'✅ Colonne "{col}" suit une loi normale → Standardisation')
            scaler = StandardScaler()
            col_num[col] = scaler.fit_transform(col_num[[col]])
        else:
            print(f'⚠️ Colonne "{col}" ne suit pas une loi normale → Normalisation MinMax')
            scaler = MinMaxScaler()
            col_num[col] = scaler.fit_transform(col_num[[col]])
            
def transformer_temps():
    """
    Transforme les colonnes temporelles (date/heure) en caractéristiques dérivées.
    """
    for col in col_temp.columns:
        if col not in df.columns:
            print(f"⚠️ La colonne '{col}' n'existe pas dans le DataFrame.")
            continue

        col_temp[col] = pd.to_datetime(col_temp[col], errors='coerce')

        # --- Colonne contenant "date" ---
        if 'date' in col.lower():
            col_temp[f'{col}_jour'] = col_temp[col].dt.day
            col_temp[f'{col}_mois'] = col_temp[col].dt.month
            col_temp[f'{col}_annee'] = col_temp[col].dt.year
            col_temp[f'{col}_jour_semaine'] = col_temp[col].dt.dayofweek
            col_temp[f'{col}_weekend'] = (col_temp[col].dt.dayofweek >= 5).astype(int)
            col_temp.drop(columns=[col], inplace=True)
            print('Bien passe')

        # --- Colonne contenant "heure" ---
        elif 'heure' in col.lower():
            col_temp[f'{col}_heure'] = col_temp[col].dt.hour
            col_temp[f'{col}_minute'] = col_temp[col].dt.minute
            col_temp[f'{col}_seconde'] = col_temp[col].dt.second
            print('Bien passe')

            # Déterminer la période de la journée
            def get_period(hour):
                if 6 <= hour < 12:
                    return 1  # matin
                elif 12 <= hour < 18:
                    return 2  # après-midi
                elif 18 <= hour < 23:
                    return 3  # soir
                else:
                    return 4  # nuit

            col_temp[f'{col}_periode'] = col_temp[f'{col}_heure'].apply(get_period)
            col_temp.drop(columns=[col], inplace=True)

        else:
            print(f"⚠️ La colonne '{col}' n'est pas reconnue comme date/heure.")


def choixEncodage():
    """
    Encode automatiquement les colonnes catégorielles :
    - Binaire si n_unique == 2
    - One-Hot si n_unique <= OneHot_threshold
    - Frequency si n_unique > OneHot_threshold

    Args:
        col_cat : DataFrame des colonnes catégorielles
        unique_cat_ratio : dict[col] = {valeur: ratio}
        OneHot_threshold : seuil max de modalités pour One-Hot
    Returns:
        df_binaire_list, df_onehot_list, df_freq_list
    """
    OneHot_threshold=10

    df_binaire_list = pd.DataFrame()
    df_onehot_list = pd.DataFrame()
    df_freq_list = pd.DataFrame()

    for col in col_cat.columns:
        n_unique = len(unique_cat_ratio[col])

        if n_unique == 2:
            # Binaire Encoding
            print(f'{col} -> Binaire')
            vals = list(unique_cat_ratio[col].keys())
            df_binaire_list[col] = col_cat[col].map({vals[0]: 0, vals[1]: 1})

        elif n_unique <= OneHot_threshold:
            # One-Hot Encoding
            print(f'{col} -> One-Hot')
            dummies = pd.get_dummies(col_cat[col], prefix=col, drop_first=False)
            df_onehot_list = pd.concat([df_onehot_list, dummies], axis=1)

        else:
            # Frequency Encoding
            print(f'{col} -> Frequency')
            freq = col_cat[col].value_counts() / len(col_cat)
            df_freq_list[col] = col_cat[col].map(freq)

    return df_binaire_list, df_onehot_list, df_freq_list

#Preproccesing de DATA
#Sppression de 30% de Données null
suppression()
#Imputation de données numérique
imputationNum()
#Imputation de données catégorique
imputationCat()
#Imputation de données temporelle:
imputationTemp(col_temp)

transformer_temps()
#Imputation de données temporelle:
imputationTemp(col_temp)

choixStdNor()

df_binaire_list, df_onehot_list, df_freq_list = choixEncodage()

final = pd.DataFrame()
def ajouter(df):
  for col in df.columns:
    final[col]= df[col]
    print(f'{col} à ajouté')
    
ajouter(col_num)
ajouter(col_temp)
ajouter(df_binaire_list)
ajouter(df_onehot_list)
ajouter(df_freq_list)
final[tagert] = df[tagert]

def feature_selection(df, seuil=0.0):
    """
    Supprime les colonnes dont la corrélation avec la variable cible est <= seuil.
    Affiche pour chaque colonne sa corrélation avant suppression.

    Args:
        df : pd.DataFrame - DataFrame contenant features et target

        seuil : float - seuil minimal de corrélation (colonnes avec corr <= seuil sont supprimées)

    Returns:
        df_filtered : pd.DataFrame - DataFrame après suppression des colonnes
        cols_removed : list - liste des colonnes supprimées
    """
    if tagert not in df.columns:
        raise ValueError(f"Variable cible '{tagert}' non trouvée dans le DataFrame")

    corr_with_tagert = df.corr()[tagert]

    cols_to_drop = []

    print(f"\nCorrélations avec la variable cible '{tagert}' :")
    print("-"*50)

    for col, corr_val in corr_with_tagert.items():
        if col == tagert:
            continue
        print(f"{col:25s} → corrélation = {corr_val:.3f} ; seuil = {seuil}")
        if corr_val <= seuil:
            cols_to_drop.append(col)

    # Supprimer les colonnes
    df_filtered = df.drop(columns=cols_to_drop)

    print("\nColonnes supprimées :")
    print(cols_to_drop)

    return df_filtered, cols_to_drop

# Appliquer la sélection de caractéristiques
final = feature_selection(final, seuil=0.0)[0]


#Algo KNN:
x = final.drop(columns=[tagert])
y = final[tagert]

rm = RandomOverSampler()
x_rm ,y_rm = rm.fit_resample(x,y)
print('old data setshape : {}'.format(Counter(y)))
print('new data setshape : {}'.format(Counter(y_rm)))

x_train,x_test,y_train,y_test = train_test_split(x_rm,y_rm,test_size=0.2,random_state=41)

def KNN(K, X_train, X_test, Y_train, Y_test):
    """
    Implémentation complète de l’algorithme KNN (K-Nearest Neighbors)
    ---------------------------------------------------------------
    K : nombre de voisins à considérer
    X_train : données d’entraînement (numpy array ou liste)
    X_test  : données de test
    Y_train : étiquettes d’entraînement
    Y_test  : étiquettes de test
    """

    # Assurer la conversion en numpy arrays
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    Y_train = np.array(Y_train)

    y_pred = []  # Liste des prédictions

    # Boucle sur chaque point de test
    for point_test in X_test:
        # Calcul des distances euclidiennes
        distances = np.sqrt(np.sum((X_train - point_test)**2, axis=1))

        # Indices triés (du plus proche au plus éloigné)
        indices_tries = np.argsort(distances)

        # Sélection des K plus proches voisins
        k_plus_proche = indices_tries[:K]

        # Détermination de la classe majoritaire
        classes = [Y_train[i] for i in k_plus_proche]
        classe_predite = Counter(classes).most_common(1)[0][0]
        y_pred.append(classe_predite)

    # Calcul des métriques de performance
    accuracy = np.mean(Y_test == y_pred)
    matrix_conf = confusion_matrix(Y_test, y_pred)
    rappel = recall_score(Y_test, y_pred, average='macro')
    f1 = f1_score(Y_test, y_pred, average='macro')

    return accuracy, matrix_conf, rappel, f1

# Tester l'algorithme KNN avec K=3
k = 3
accuracy, matrix_conf, rappel, f1 = KNN(k, x_train, x_test, y_train, y_test)
print(f"Accuracy : {accuracy}")
print(f"Matrice de confusion : \n{matrix_conf}")
print(f"Rappel : {rappel}")
print(f"F1-score : {f1}")