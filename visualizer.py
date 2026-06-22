# ============================================================
# visualizer.py  —  8 Professional Charts (Myntra Dataset)
# ============================================================
import os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

OUTPUT_DIR = '/tmp/fashion_ai_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROSE='#C9706A'; NAVY='#3B5070'; SAGE='#7A9E7E'; AMBER='#D4A853'
LILAC='#8F7BB5'; TEAL='#4E9E9B'; CORAL='#E8866A'; SLATE='#607080'
CREAM='#F5F0E8'; DARK='#2C2C2C'
PALETTE=[ROSE,NAVY,SAGE,AMBER,LILAC,TEAL,CORAL,SLATE,'#9E6A8F','#6A8F9E']

sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#FAFAFA'})

def _save(fig, fn):
    p = os.path.join(OUTPUT_DIR, fn)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [CHART] {fn}")
    return p

# ── 1. Style Tag Distribution ────────────────────────────────
def plot_class_distribution(df):
    counts  = df['style_tag'].value_counts()
    explode = [0.04]*len(counts)
    fig, ax = plt.subplots(figsize=(9,9))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=PALETTE[:len(counts)], explode=explode, startangle=130,
        textprops={'fontsize':11,'color':DARK},
        wedgeprops={'linewidth':1.5,'edgecolor':'white'})
    for at in autotexts:
        at.set_fontweight('bold'); at.set_fontsize(10)
    ax.set_title('Style Tag Distribution — 10 Classes\n(Target variable the ML model predicts)',
                 fontsize=14,fontweight='bold',color=DARK,pad=20)
    fig.tight_layout()
    return _save(fig,'1_class_distribution.png')

# ── 2. Feature Heatmaps ──────────────────────────────────────
def plot_feature_heatmap(df):
    fig, axes = plt.subplots(1,3,figsize=(18,6))
    fig.suptitle('Feature → Style Tag Association Heatmaps\n(darker cell = more products of that style for that feature value)',
                 fontsize=13,fontweight='bold',color=DARK,y=1.02)
    for ax, feat in zip(axes,['body_type','skin_tone','occasion']):
        pivot = pd.crosstab(df[feat], df['style_tag'])
        sns.heatmap(pivot,annot=True,fmt='d',cmap='RdPu',linewidths=0.5,
                    linecolor='white',ax=ax,cbar=False,annot_kws={'size':8})
        ax.set_title(feat.replace('_',' ').title(),fontsize=12,fontweight='bold',color=DARK)
        ax.set_xlabel('Style Tag',fontsize=9); ax.set_ylabel(feat.replace('_',' ').title(),fontsize=9)
        ax.tick_params(axis='x',rotation=40,labelsize=7)
        ax.tick_params(axis='y',rotation=0,labelsize=9)
    fig.tight_layout()
    return _save(fig,'2_feature_heatmap.png')

# ── 3. Price Distribution by Style Tag ───────────────────────
def plot_price_distribution(df):
    if 'price_inr' not in df.columns:
        return ''
    order  = df.groupby('style_tag')['price_inr'].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(13,6))
    sns.boxplot(data=df, x='style_tag', y='price_inr', order=order,
                palette=PALETTE[:len(order)], ax=ax,
                flierprops={'marker':'o','markersize':3,'alpha':0.4})
    ax.set_title('Price Distribution (INR) by Style Tag\n(Real Myntra-style pricing)',
                 fontsize=13,fontweight='bold',color=DARK,pad=15)
    ax.set_xlabel('Style Tag',fontsize=11); ax.set_ylabel('Price (INR)',fontsize=11)
    ax.tick_params(axis='x',rotation=30,labelsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{int(x):,}'))
    fig.tight_layout()
    return _save(fig,'3_price_distribution.png')

# ── 4. Model Accuracy Comparison ─────────────────────────────
def plot_model_comparison(results):
    names=list(results.keys())
    train_accs=[results[n]['train_acc']*100 for n in names]
    test_accs =[results[n]['test_acc'] *100 for n in names]
    x=np.arange(len(names)); w=0.35
    fig,ax=plt.subplots(figsize=(9,5))
    b1=ax.bar(x-w/2,train_accs,w,label='Train Accuracy',color=NAVY,alpha=0.85,edgecolor='white',linewidth=1)
    b2=ax.bar(x+w/2,test_accs, w,label='Test Accuracy', color=ROSE,alpha=0.85,edgecolor='white',linewidth=1)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.8,
                f"{bar.get_height():.1f}%",ha='center',va='bottom',fontsize=10,fontweight='bold',color=DARK)
    ax.set_title('Model Accuracy Comparison: Train vs Test',fontsize=13,fontweight='bold',color=DARK,pad=15)
    ax.set_ylabel('Accuracy (%)',fontsize=11); ax.set_xticks(x)
    ax.set_xticklabels(names,fontsize=11); ax.set_ylim(0,110)
    ax.legend(fontsize=10,framealpha=0.5)
    best_test=max(test_accs)
    ax.annotate(f'Best test: {best_test:.1f}%\n(random = {100//len(results[names[0]]["cm"])}%)',
                xy=(0.97,0.95),xycoords='axes fraction',ha='right',va='top',fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4',facecolor=CREAM,edgecolor=SLATE,alpha=0.8))
    fig.tight_layout()
    return _save(fig,'4_model_comparison.png')

# ── 5. Confusion Matrix ──────────────────────────────────────
def plot_confusion_matrix(cm, class_names, model_name):
    fig,ax=plt.subplots(figsize=(11,9))
    cm_norm=cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-9)
    im=ax.imshow(cm_norm,cmap='Blues',vmin=0,vmax=1)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label='Row-normalised proportion')
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names,rotation=45,ha='right',fontsize=9)
    ax.set_yticklabels(class_names,fontsize=9)
    thresh=0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color='white' if cm_norm[i,j]>thresh else DARK
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,fontweight='bold',color=color)
    ax.set_title(f'Confusion Matrix — {model_name}\n(diagonal = correct predictions)',
                 fontsize=13,fontweight='bold',color=DARK,pad=15)
    ax.set_ylabel('True Label',fontsize=11); ax.set_xlabel('Predicted Label',fontsize=11)
    fig.tight_layout()
    return _save(fig,'5_confusion_matrix.png')

# ── 6. Feature Importance ────────────────────────────────────
def plot_feature_importance(importance_dict):
    if not importance_dict: return ''
    items=sorted(importance_dict.items(),key=lambda x:x[1])
    feats=[i[0].replace('_',' ').title() for i in items]
    vals =[i[1]*100 for i in items]
    colors=[ROSE if v==max(vals) else NAVY for v in vals]
    fig,ax=plt.subplots(figsize=(8,5))
    bars=ax.barh(feats,vals,color=colors,edgecolor='white',linewidth=1,height=0.55)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_width()+0.4,bar.get_y()+bar.get_height()/2,
                f'{val:.1f}%',va='center',fontsize=11,fontweight='bold',color=DARK)
    ax.set_title('Feature Importance (Random Forest)\nWhich input drives the style prediction most?',
                 fontsize=13,fontweight='bold',color=DARK,pad=15)
    ax.set_xlabel('Importance (%)',fontsize=11); ax.set_xlim(0,max(vals)*1.25)
    ax.legend(handles=[mpatches.Patch(color=ROSE,label='Most important'),
                       mpatches.Patch(color=NAVY,label='Others')],fontsize=9,loc='lower right')
    fig.tight_layout()
    return _save(fig,'6_feature_importance.png')

# ── 7. Precision / Recall / F1 ───────────────────────────────
def plot_precision_recall_f1(y_test, y_pred, class_names, model_name):
    report=classification_report(y_test,y_pred,target_names=class_names,output_dict=True,zero_division=0)
    classes=[c for c in class_names if c in report]
    precision=[report[c]['precision']*100 for c in classes]
    recall   =[report[c]['recall']   *100 for c in classes]
    f1       =[report[c]['f1-score'] *100 for c in classes]
    x=np.arange(len(classes)); w=0.26
    fig,ax=plt.subplots(figsize=(14,5))
    ax.bar(x-w,  precision,w,label='Precision',color=NAVY, alpha=0.85,edgecolor='white')
    ax.bar(x,    recall,   w,label='Recall',   color=ROSE, alpha=0.85,edgecolor='white')
    ax.bar(x+w,  f1,       w,label='F1-Score', color=SAGE, alpha=0.85,edgecolor='white')
    ax.set_title(f'Precision / Recall / F1-Score per Class — {model_name}',
                 fontsize=13,fontweight='bold',color=DARK,pad=15)
    ax.set_ylabel('Score (%)',fontsize=11); ax.set_xticks(x)
    ax.set_xticklabels(classes,rotation=30,ha='right',fontsize=9); ax.set_ylim(0,120)
    macro_f1=report['macro avg']['f1-score']*100
    ax.axhline(y=macro_f1,color=AMBER,linestyle='--',linewidth=1.8,alpha=0.9)
    ax.text(len(classes)-0.5,macro_f1+2,f'Macro F1: {macro_f1:.1f}%',
            fontsize=9,color=AMBER,fontweight='bold')
    ax.legend(fontsize=10,loc='upper right',framealpha=0.5)
    fig.tight_layout()
    return _save(fig,'7_precision_recall_f1.png')

# ── 8. Prediction Confidence ─────────────────────────────────
def plot_prediction_confidence(probabilities, user_name, predicted_style):
    top=sorted(probabilities.items(),key=lambda x:x[1],reverse=True)[:8]
    labels=[t[0] for t in top]; values=[t[1]*100 for t in top]
    colors=[ROSE if lbl==predicted_style else '#CCCCCC' for lbl in labels]
    fig,ax=plt.subplots(figsize=(10,5))
    bars=ax.bar(labels,values,color=colors,edgecolor='white',linewidth=1,width=0.6)
    for bar,val in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.6,
                f'{val:.0f}%',ha='center',va='bottom',fontsize=11,fontweight='bold',color=DARK)
    ax.set_title(f'Prediction Confidence for {user_name}\nModel predicts: "{predicted_style.upper().replace("_"," ")}" style',
                 fontsize=13,fontweight='bold',color=DARK,pad=15)
    ax.set_ylabel('Confidence (%)',fontsize=11); ax.set_ylim(0,max(values)*1.3+5)
    ax.tick_params(axis='x',labelsize=10)
    ax.legend(handles=[mpatches.Patch(color=ROSE,label=f'Predicted: {predicted_style}'),
                       mpatches.Patch(color='#CCCCCC',label='Other styles')],fontsize=9,loc='upper right')
    fig.tight_layout()
    return _save(fig,'8_prediction_confidence.png')

# ── MASTER ───────────────────────────────────────────────────
def generate_all_charts(results, best_name, df, feature_importance, class_names,
                        y_test=None, y_pred=None, probabilities=None,
                        user_name='User', predicted_style=''):
    print('\n[CHARTS] Generating all visualisations...')
    paths=[]
    paths.append(plot_class_distribution(df))
    paths.append(plot_feature_heatmap(df))
    paths.append(plot_price_distribution(df))
    paths.append(plot_model_comparison(results))
    paths.append(plot_confusion_matrix(results[best_name]['cm'],class_names,best_name))
    paths.append(plot_feature_importance(feature_importance))
    if y_test is not None and y_pred is not None:
        paths.append(plot_precision_recall_f1(y_test,y_pred,class_names,best_name))
    if probabilities:
        paths.append(plot_prediction_confidence(probabilities,user_name,predicted_style))
    paths=[p for p in paths if p]
    print(f'[CHARTS] {len(paths)} charts saved to /outputs/')
    return paths

# ── 9. Budget vs Available Outfits ───────────────────────────
def plot_budget_analysis(df: pd.DataFrame, min_price: int, max_price: int,
                         style_tag: str, result: dict) -> str:
    """
    Shows where the user's budget sits relative to the full
    price distribution for their predicted style.
    """
    style_df = df[df['style_tag'] == style_tag]['price_inr']
    if style_df.empty:
        return ''

    fig, ax = plt.subplots(figsize=(10, 5))

    # Full price histogram for this style
    ax.hist(style_df, bins=20, color=NAVY, alpha=0.6, edgecolor='white',
            linewidth=0.8, label=f'All {style_tag.replace("_"," ").title()} outfits')

    # Shade the user's budget range
    ax.axvspan(min_price, max_price, alpha=0.25, color=ROSE,
               label=f'Your budget ₹{min_price:,}–₹{max_price:,}')

    # Budget boundary lines
    ax.axvline(min_price, color=ROSE, linewidth=2, linestyle='--')
    ax.axvline(max_price, color=ROSE, linewidth=2, linestyle='--')

    # Count in range
    in_range_count = len(result['in_range']) if not result['in_range'].empty else 0
    total_count    = len(style_df)

    ax.set_title(f'Budget Analysis — {style_tag.replace("_"," ").title()} Style\n'
                 f'{in_range_count}/{total_count} outfits fall within your ₹{min_price:,}–₹{max_price:,} budget',
                 fontsize=13, fontweight='bold', color=DARK, pad=15)
    ax.set_xlabel('Price (INR)', fontsize=11)
    ax.set_ylabel('Number of Outfits', fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{int(x):,}'))
    ax.legend(fontsize=10, framealpha=0.6)

    # Annotation
    pct = in_range_count / total_count * 100 if total_count else 0
    ax.annotate(f'{pct:.0f}% of {style_tag.replace("_"," ")}\noutfits in budget',
                xy=((min_price+max_price)/2, ax.get_ylim()[1]*0.85),
                ha='center', fontsize=10, fontweight='bold',
                color=ROSE if pct < 30 else SAGE,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=CREAM,
                          edgecolor=SLATE, alpha=0.8))

    fig.tight_layout()
    return _save(fig, '9_budget_analysis.png')
