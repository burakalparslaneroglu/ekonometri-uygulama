"""Konu 12 için HC0--HC3, heteroskedastisite tanıları ve robust ortak çıkarım."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, f as f_dist, t as student_t
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from core.joint_inference_utils import LinearRestriction
from core.regression_inference_utils import OLSInferenceResult, fit_ols_inference

CovarianceType=Literal["HC0","HC1","HC2","HC3"]
_COVARIANCE_TYPES=("HC0","HC1","HC2","HC3")

@dataclass(frozen=True)
class RobustInferenceResult:
    """OLS nokta tahminlerini ve seçilmiş HC kovaryansındaki çıkarımı taşır."""
    ols: OLSInferenceResult; covariance_type: str; coefficients: pd.Series; standard_errors: pd.Series; t_values_zero: pd.Series
    p_values_two_sided_zero: pd.Series; confidence_intervals_95: pd.DataFrame; covariance_matrix: pd.DataFrame
    def __post_init__(self)->None:
        for name in ("coefficients","standard_errors","t_values_zero","p_values_two_sided_zero"):
            object.__setattr__(self,name,getattr(self,name).copy(deep=True))
        object.__setattr__(self,"confidence_intervals_95",self.confidence_intervals_95.copy(deep=True)); object.__setattr__(self,"covariance_matrix",self.covariance_matrix.copy(deep=True))

@dataclass(frozen=True)
class HeteroskedasticityTest:
    """LM testi ve açıklayıcı karar metni."""
    name:str; lm_statistic:float; p_value:float; df:int; reject_null:bool; null_hypothesis:str

@dataclass(frozen=True)
class RobustJointTest:
    """Seçilmiş HC kovaryansıyla Wald/F biçimli ortak test."""
    covariance_type:str; q:int; f_statistic:float; p_value:float; df_denom:int; reject_null:bool; restrictions:tuple[str,...]

@dataclass(frozen=True)
class SimulationSummary:
    """Heteroskedastik DGP altında vektörize kapsama benzetimi özeti."""
    slope_mean:float; empirical_slope_sd:float; mean_nonrobust_se:float; mean_hc1_se:float; mean_hc3_se:float
    nonrobust_coverage:float; hc1_coverage:float; hc3_coverage:float; repetitions:int; seed:int

@dataclass(frozen=True)
class BreuschPaganDetails:
    """BP yardımcı regresyonunun yeniden üretilebilir ayrıntıları."""
    auxiliary_variables:tuple[str,...]; auxiliary_r_squared:float; nobs:int; lm_manual:float
    lm_statsmodels:float; lm_p_value:float; f_statistic:float; f_p_value:float; df:int

@dataclass(frozen=True)
class WhiteTestDetails:
    """White yardımcı tasarımının terimleri, rank'ı ve LM ayrıntıları."""
    base_variables:tuple[str,...]; square_variables:tuple[str,...]; interaction_variables:tuple[str,...]
    auxiliary_columns:tuple[str,...]; matrix_rank:int; column_count:int; auxiliary_r_squared:float
    nobs:int; lm_manual:float; lm_statsmodels:float; p_value:float; df:int

def _validate_covariance(covariance_type:str)->CovarianceType:
    if covariance_type not in _COVARIANCE_TYPES: raise ValueError("Kovaryans türü HC0, HC1, HC2 veya HC3 olmalıdır.")
    return covariance_type # type: ignore[return-value]

def fit_robust_inference(frame:pd.DataFrame,dependent:str,explanatory:tuple[str,...],*,covariance_type:str="HC1")->RobustInferenceResult:
    """Aynı OLS katsayılarıyla ayrı HC çıkarım sonucu üretir."""
    cov=_validate_covariance(covariance_type); ols=fit_ols_inference(frame,dependent,explanatory)
    design=sm.add_constant(ols.design_data,has_constant="add"); robust=sm.OLS(ols.observed_values,design).fit().get_robustcov_results(cov_type=cov,use_t=True)
    names=list(ols.coefficients.index); params=pd.Series(robust.params,index=names); se=pd.Series(robust.bse,index=names); tvals=pd.Series(robust.tvalues,index=names); pvals=pd.Series(robust.pvalues,index=names)
    intervals=pd.DataFrame(robust.conf_int(),index=names,columns=["lower","upper"]); matrix=pd.DataFrame(robust.cov_params(),index=names,columns=names)
    if not np.allclose(params.to_numpy(),ols.coefficients.to_numpy()): raise ValueError("HC hesabı OLS katsayılarını değiştirmemelidir.")
    return RobustInferenceResult(ols,cov,params,se,tvals,pvals,intervals,matrix)

def heteroskedasticity_tests(result:OLSInferenceResult,*,alpha:float=.05)->tuple[HeteroskedasticityTest,HeteroskedasticityTest]:
    """Breusch--Pagan ve White LM tanılarını aynı model örnekleminde hesaplar."""
    if not 0<alpha<1: raise ValueError("alpha 0 ile 1 arasında olmalıdır.")
    design=sm.add_constant(result.design_data,has_constant="add"); bp=het_breuschpagan(result.residuals,design); white=het_white(result.residuals,design)
    null="H0: koşullu hata varyansı sabittir (homoskedastisite)."
    return (HeteroskedasticityTest("Breusch–Pagan",float(bp[0]),float(bp[1]),design.shape[1]-1,bool(bp[1]<alpha),null),HeteroskedasticityTest("White",float(white[0]),float(white[1]),int(round(white[2]*0+len(design.columns)*(len(design.columns)+1)/2-1)),bool(white[1]<alpha),null))

def robust_joint_test(result:RobustInferenceResult,restrictions:tuple[LinearRestriction,...],*,alpha:float=.05)->RobustJointTest:
    """Robust kovaryans ile Rb=r Wald/F testini kurar; nonrobust API'yi değiştirmez."""
    if not restrictions or not 0<alpha<1: raise ValueError("En az bir kısıt ve geçerli alpha gerekir.")
    names=list(result.ols.coefficients.index); rows=[]; rhs=[]; labels=[]
    for restriction in restrictions:
        unknown=set(restriction.weights).difference(names)
        if unknown: raise ValueError(f"Modelde olmayan katsayı: {', '.join(sorted(unknown))}")
        row=np.array([float(restriction.weights.get(name,0.0)) for name in names]); rows.append(row); rhs.append(float(restriction.rhs)); labels.append(restriction.label)
    R=np.vstack(rows); q=int(np.linalg.matrix_rank(R))
    if q != len(rows): raise ValueError("Kısıtlar bağımsız olmalıdır.")
    diff=R@result.ols.coefficients.to_numpy(float)-np.asarray(rhs); middle=R@result.covariance_matrix.to_numpy(float)@R.T
    if np.linalg.matrix_rank(middle)<q: raise ValueError("Kısıt kovaryans matrisi tekildir.")
    statistic=float(diff.T@np.linalg.solve(middle,diff)/q); p=float(f_dist.sf(statistic,q,result.ols.df_resid))
    return RobustJointTest(result.covariance_type,q,statistic,p,result.ols.df_resid,bool(p<alpha),tuple(labels))

def residual_plot_data(result:OLSInferenceResult)->pd.DataFrame:
    """Artık-tahmin ve studentize ölçek-konum grafikleri için sonlu veri hazırlar."""
    fitted=result.fitted_values.to_numpy(float)
    residuals=result.residuals.to_numpy(float)
    design=sm.add_constant(result.design_data,has_constant="add").to_numpy(float)
    inverse=np.linalg.inv(design.T@design)
    leverage=np.einsum("ij,jk,ik->i",design,inverse,design)
    denominator=max(result.residual_standard_deviation,1e-12)*np.sqrt(np.maximum(1.0-leverage,1e-12))
    studentized=residuals/denominator
    scale=np.sqrt(np.abs(studentized))
    if not np.isfinite(np.column_stack((fitted,residuals,studentized,scale))).all(): raise ValueError("Grafik verisi sonlu olmalıdır.")
    return pd.DataFrame({"tahmin":fitted,"artık":residuals,"mutlak_artık":np.abs(residuals),"studentize_artık":studentized,"scale_location":scale})

def simulate_heteroskedastic_coverage(*,nobs:int=60,repetitions:int=4000,seed:int=202512)->SimulationSummary:
    """Binlerce statsmodels uyumu yerine vektörize HC0--HC3 kapsama benzetimi yürütür."""
    if nobs<10 or repetitions<10: raise ValueError("nobs ve repetitions yeterince büyük olmalıdır.")
    rng=np.random.default_rng(seed); x=rng.uniform(0,4,size=(repetitions,nobs)); X=np.stack((np.ones_like(x),x),axis=2); errors=rng.normal(size=(repetitions,nobs))*(.3+.7*x**2); y=1+2*x+errors
    inv=np.linalg.inv(np.einsum("rni,rnj->rij",X,X)); beta=np.einsum("rij,rnj,rn->ri",inv,X,y); residual=y-np.einsum("rni,ri->rn",X,beta)
    leverage=np.einsum("rni,rij,rnj->rn",X,inv,X); xtx=np.einsum("rni,rnj->rij",X,X); sigma2=np.sum(residual**2,axis=1)/(nobs-2); non=np.sqrt(sigma2*inv[:,1,1])
    def hc(kind:str)->np.ndarray:
        if kind=="HC1": weights=residual**2*nobs/(nobs-2)
        elif kind=="HC3": weights=residual**2/(1-leverage)**2
        else: weights=residual**2
        meat=np.einsum("rni,rn,rnj->rij",X,weights,X); cov=np.einsum("rij,rjk,rkl->ril",inv,meat,inv); return np.sqrt(np.maximum(cov[:,1,1],0))
    hc1,hc3=hc("HC1"),hc("HC3"); critical=float(student_t.ppf(.975,nobs-2)); slope=beta[:,1]
    return SimulationSummary(float(slope.mean()),float(slope.std(ddof=1)),float(non.mean()),float(hc1.mean()),float(hc3.mean()),float(np.mean(np.abs(slope-2)<=critical*non)),float(np.mean(np.abs(slope-2)<=critical*hc1)),float(np.mean(np.abs(slope-2)<=critical*hc3)),repetitions,seed)


def breusch_pagan_auxiliary_details(result:OLSInferenceResult)->BreuschPaganDetails:
    """Artık kareleri yardımcı regresyonundan BP LM=nR² hesabını açar."""
    design=sm.add_constant(result.design_data,has_constant="add")
    auxiliary=sm.OLS(result.residuals.to_numpy(float)**2,design).fit()
    bp=het_breuschpagan(result.residuals,design)
    manual=float(result.nobs*auxiliary.rsquared)
    return BreuschPaganDetails(tuple(result.design_data.columns),float(auxiliary.rsquared),result.nobs,manual,float(bp[0]),float(bp[1]),float(bp[2]),float(bp[3]),len(result.design_data.columns))


def white_auxiliary_design(result:OLSInferenceResult)->tuple[pd.DataFrame,tuple[str,...],tuple[str,...]]:
    """White testi için düzey, kare ve çapraz terimleri deterministik adlarla kurar."""
    base=result.design_data.apply(pd.to_numeric,errors="coerce")
    design=pd.DataFrame(index=base.index); design["const"]=1.0
    for name in base.columns: design[name]=base[name]
    squares=[]; interactions=[]; names=list(base.columns)
    for index,name in enumerate(names):
        square=f"{name}²"; design[square]=base[name]**2; squares.append(square)
        for other in names[index+1:]:
            interaction=f"{name}×{other}"; design[interaction]=base[name]*base[other]; interactions.append(interaction)
    return design,tuple(squares),tuple(interactions)


def white_test_details(result:OLSInferenceResult)->WhiteTestDetails:
    """White LM testini açık yardımcı tasarım ve manuel nR² ile üretir."""
    design,squares,interactions=white_auxiliary_design(result)
    independent=[]; current=np.empty((len(design),0))
    for name in design.columns:
        candidate=np.column_stack((current,design[name].to_numpy(float)))
        if np.linalg.matrix_rank(candidate)>np.linalg.matrix_rank(current): independent.append(name); current=candidate
    reduced=design.loc[:,independent]
    auxiliary=sm.OLS(result.residuals.to_numpy(float)**2,reduced).fit()
    white=het_white(result.residuals,sm.add_constant(result.design_data,has_constant="add"))
    rank=int(np.linalg.matrix_rank(reduced.to_numpy(float))); df=rank-1
    return WhiteTestDetails(tuple(result.design_data.columns),squares,interactions,tuple(reduced.columns),rank,reduced.shape[1],float(auxiliary.rsquared),result.nobs,float(result.nobs*auxiliary.rsquared),float(white[0]),float(white[1]),df)


def coefficient_interval_comparison(frame:pd.DataFrame,dependent:str,explanatory:tuple[str,...])->pd.DataFrame:
    """Geleneksel ve HC0–HC3 çıkarımını tek doğrulanmış tabloda birleştirir."""
    conventional=fit_ols_inference(frame,dependent,explanatory)
    table=pd.DataFrame({"coefficient":conventional.coefficients.index,"estimate":conventional.coefficients.values,"nonrobust_se":conventional.standard_errors.values,"nonrobust_p":conventional.p_values_two_sided_zero.values})
    for cov in _COVARIANCE_TYPES:
        result=fit_robust_inference(frame,dependent,explanatory,covariance_type=cov)
        table[f"{cov}_se"]=result.standard_errors.values; table[f"{cov}_p"]=result.p_values_two_sided_zero.values
        table[f"{cov}_lower"]=result.confidence_intervals_95["lower"].values; table[f"{cov}_upper"]=result.confidence_intervals_95["upper"].values
    return table


def heteroskedastic_pattern_simulation(*,pattern:str="artan",nobs:int=180,seed:int=202512)->pd.DataFrame:
    """Aynı koşullu ortalama altında değişen varyans örüntülerini üretir."""
    if pattern not in ("sabit","artan","azalan","U biçimli") or nobs<20:
        raise ValueError("Desteklenen örüntü ve yeterli gözlem sayısı gerekir.")
    rng=np.random.default_rng(seed); x=np.linspace(-2,2,nobs); mean=1+2*x
    scales={"sabit":np.full(nobs,.8),"artan":.25+.55*(x-x.min()),"azalan":.25+.55*(x.max()-x),"U biçimli":.3+.6*np.abs(x)}[pattern]
    y=mean+rng.normal(size=nobs)*scales
    return pd.DataFrame({"x":x,"koşullu_ortalama":mean,"y":y,"hata_sd":scales,"örüntü":pattern})


def simulation_slope_samples(*,nobs:int=60,repetitions:int=1000,seed:int=202512)->pd.DataFrame:
    """Kaynak DGP'sinden histogram için vektörize eğim örnekleri üretir."""
    if nobs<10 or repetitions<10: raise ValueError("nobs ve repetitions yeterli olmalıdır.")
    rng=np.random.default_rng(seed); x=rng.uniform(0,4,size=(repetitions,nobs)); y=1+2*x+rng.normal(size=(repetitions,nobs))*(.3+.7*x**2)
    xc=x-x.mean(axis=1,keepdims=True); slope=np.sum(xc*(y-y.mean(axis=1,keepdims=True)),axis=1)/np.sum(xc**2,axis=1)
    return pd.DataFrame({"eğim":slope})
