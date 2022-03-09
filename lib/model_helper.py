from logging import getLogger

logger = getLogger(__name__)


def incarca_parametrii_model_bun(model):
    return model.incarca(model.config.confProiect.cale_config_cel_mai_bun_model,
                         model.config.confProiect.cale_parametrii_cel_mai_bun_model)


def salveaza_ca_cel_mai_bun_model(model):
    return model.salveaza(model.config.confProiect.cale_config_cel_mai_bun_model,
                          model.config.confProiect.cale_parametrii_cel_mai_bun_model)


def testeaza_schimbari(model):
    logger.debug("Testare schimbare model")
    digest = model.fetch_digest(model.config.confProiect.cale_parametrii_cel_mai_bun_model)
    if digest != model.digest:
        return incarca_parametrii_model_bun(model)
    logger.debug("Cel mai bun model nu a fost schimbat cu un candidat")
    return False
