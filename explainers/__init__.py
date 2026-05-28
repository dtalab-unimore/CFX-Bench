import explainers.nice_actionables_explainer
from explainers.ar_explainer import ARExplainer
from explainers.bfcf_explainer import BruteForceCounterfactualExplainer
from explainers.dice_explainer import DiceExplainer
from explainers.face_explainer import FaceExplainer
from explainers.global_explainers.ares_explainer import AresExplainer
from explainers.nice_actionables_explainer import NiceActionablesExplainer
from explainers.nice_explainer import NiceExplainer
from explainers.optbin_explainer import OptBinExplainer
from explainers.proce_explainer import ProCEAEBinnedExplainer


def get_cf_explainer(expl_name, expl_params):
    if expl_name == 'ar':
        return ARExplainer(**expl_params)
    if expl_name == 'optbin':
        return OptBinExplainer(**expl_params)
    if expl_name == 'nice':
        return NiceExplainer(**expl_params)
    if expl_name == 'nice-actionables':
        return NiceActionablesExplainer(**expl_params)
    if expl_name == 'dice':
        return DiceExplainer(**expl_params)
    if expl_name == 'face':
        return FaceExplainer(**expl_params)
    if expl_name == 'bfcf':
        return BruteForceCounterfactualExplainer(**expl_params)
    if expl_name == 'proce':
        return ProCEAEBinnedExplainer(**expl_params)

    if expl_name == 'ares':
        return AresExplainer(**expl_params)

    raise ValueError("Explainer not found!")
