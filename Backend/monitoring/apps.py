from django.apps import AppConfig


class MonitoringConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'monitoring'

    def ready(self):
        #ensure Mongo connects when Django starts
        from atmospheric_gases.mongo import init_mongo
        init_mongo()
