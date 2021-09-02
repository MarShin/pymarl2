from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner
from .lica_learner import LICALearner
from .nq_learner import NQLearner
from .policy_gradient_v2 import PGLearner_v2
from .max_q_learner import MAXQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .offpg_learner import OffPGLearner
from .maven_learner import MAVENLearner
from .fmac_learner import FMACLearner
from .vannila_q_learner import VanillaQLearner

REGISTRY = {}

REGISTRY["vanilla_q_learner"] = VanillaQLearner
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["lica_learner"] = LICALearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["policy_gradient_v2"] = PGLearner_v2
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["maven_learner"] = MAVENLearner
REGISTRY["fmac_learner"] = FMACLearner
