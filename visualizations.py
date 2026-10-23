import plotly.express as px
import plotly.graph_objects as go
from config import MODEL_LABELS
COLORS={"Bicubic":"#9aaac3","Bilinear":"#648cff","CNN_A":"#42d9f5","CNN_B":"#f2b866"}
def comparison_chart(df,metric,title=None):
    data=df[df.model.isin(MODEL_LABELS)].copy()
    return px.bar(data,x="model",y=metric,color="model",color_discrete_map=COLORS,
      labels={"model":"Method",metric:metric.replace("_"," ")},title=title or metric.replace("_"," "),
      template="plotly_dark",text_auto=".4f").update_layout(showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
def heatmap(a,title="Elevation"):
    fig=go.Figure(go.Heatmap(z=a,colorscale="Gray",colorbar={"title":"elevation"}))
    return fig.update_layout(title=title,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
def training_chart(df,mode,seed):
    d=df[(df["mode"]==mode)&(df.seed==seed)].sort_values("epoch")
    fig=go.Figure()
    for col,label in [("train_elevation_l1","Training L1"),("val_elevation_l1","Validation L1")]:
        fig.add_scatter(x=d.epoch,y=d[col],mode="lines",name=label)
    return fig.update_layout(title=f"CNN-{mode} · seed {seed}",xaxis_title="Epoch",yaxis_title="Patch-centered elevation L1",template="plotly_dark")
