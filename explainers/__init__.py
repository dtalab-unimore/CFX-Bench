def get_cf_explainer(expl_name, expl_params):
    if expl_name == 'ar':
        from explainers.ar_explainer import ARExplainer
        return ARExplainer(**expl_params)
    if expl_name == 'optbin':
        from explainers.optbin_explainer import OptBinExplainer
        return OptBinExplainer(**expl_params)
    if expl_name == 'nice':
        from explainers.nice_explainer import NiceExplainer
        return NiceExplainer(**expl_params)
    if expl_name == 'dice':
        from explainers.dice_explainer import DiceExplainer
        return DiceExplainer(**expl_params)
    if expl_name == 'face':
        from explainers.face_explainer import FaceExplainer
        return FaceExplainer(**expl_params)
    if expl_name == 'proce':
        from explainers.proce_explainer import ProCEAEBinnedExplainer
        return ProCEAEBinnedExplainer(**expl_params)

    if expl_name == 'ares':
        from explainers.global_explainers.ares_explainer import AresExplainer
        return AresExplainer(**expl_params)
    if expl_name == 'globe-ce':
        from explainers.global_explainers.globece_explainer import GlobeCeExplainer
        return GlobeCeExplainer(**expl_params)
    if expl_name == 'facegroup':
        from explainers.global_explainers.facegroup_explainer import FaceGroupExplainer
        return FaceGroupExplainer(**expl_params)
    if expl_name == 'glance':
        from explainers.global_explainers.glance_explainer import GlanceExplainer
        return GlanceExplainer(**expl_params)
    if expl_name == 'llm-global':
        from explainers.global_explainers.llmglobal_explainer import LlmGlobalExplainer
        return LlmGlobalExplainer(**expl_params)
    if expl_name == 'llm-local':
        from explainers.global_explainers.llmlocal_explainer import LlmLocalExplainer
        return LlmLocalExplainer(**expl_params)

    raise ValueError("Explainer not found!")
