from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication


def get_workspace():
    return Workspace(subscription_id='83b4b5c6-51ae-4d5a-a7cf-63d20ffc2754',
                resource_group='FedWatchApp93',
                workspace_name='ltcftmlvmeibpdqx7w6a',
                auth=InteractiveLoginAuthentication(tenant_id='9c8b18d3-0d09-4525-a546-341d38f12190'))


args = {
    'runinfo': '/home/user/Documents/Azure-ML-Dashboard/data/runinfo',
    'trainlog': '/home/user/Documents/Azure-ML-Dashboard/data/trainlog',
    'workspace': get_workspace()
}