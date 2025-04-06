from django.db import models

# Create your models here.

class FloodPrediction(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    water_level = models.FloatField()
    river_discharge = models.FloatField()
    rainfall = models.FloatField()
    elevation = models.FloatField()
    humidity = models.FloatField()
    temperature = models.FloatField()
    population_density = models.FloatField()
    predicted_risk = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.timestamp} - Risk: {self.predicted_risk}"
