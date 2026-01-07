import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
} from '@mui/material';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface TemporalParams {
  predictionSteps: number;
  interventionType: 'none' | 'antibiotic' | 'probiotic' | 'diet';
}

interface TemporalData {
  timePoint: number;
  bacteroides: number;
  firmicutes: number;
  proteobacteria: number;
  actinobacteria: number;
  diversity: number;
}

interface PredictionMetrics {
  accuracy: number;
  stability: number;
}

const Temporal: React.FC = () => {
  const [params, setParams] = useState<TemporalParams>({
    predictionSteps: 5,
    interventionType: 'none',
  });

  const [isPredicting, setIsPredicting] = useState(false);
  const [temporalData, setTemporalData] = useState<TemporalData[]>([]);
  const [predictionMetrics, setPredictionMetrics] = useState<PredictionMetrics | null>(null);

  const generateMockTemporal = (steps: number, intervention: string): {
    data: TemporalData[];
    metrics: PredictionMetrics;
  } => {
    const data: TemporalData[] = [];
    
    let bacteroides = 0.4;
    let firmicutes = 0.35;
    let proteobacteria = 0.15;
    let actinobacteria = 0.1;
    
    const interventionEffects = {
      none: { bac: 0, fir: 0, pro: 0, act: 0 },
      antibiotic: { bac: -0.02, fir: -0.03, pro: 0.03, act: 0.02 },
      probiotic: { bac: 0.01, fir: 0.02, pro: -0.01, act: -0.02 },
      diet: { bac: 0.015, fir: -0.01, pro: -0.005, act: 0 },
    };
    
    const effects = interventionEffects[intervention as keyof typeof interventionEffects];
    
    for (let i = 0; i <= steps; i++) {
      if (i > 0) {
        bacteroides += effects.bac + (Math.random() - 0.5) * 0.01;
        firmicutes += effects.fir + (Math.random() - 0.5) * 0.01;
        proteobacteria += effects.pro + (Math.random() - 0.5) * 0.005;
        actinobacteria += effects.act + (Math.random() - 0.5) * 0.005;
        
        const total = bacteroides + firmicutes + proteobacteria + actinobacteria;
        bacteroides /= total;
        firmicutes /= total;
        proteobacteria /= total;
        actinobacteria /= total;
      }
      
      const abundances = [bacteroides, firmicutes, proteobacteria, actinobacteria];
      const diversity = -abundances.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
      
      data.push({
        timePoint: i,
        bacteroides: Math.round(bacteroides * 1000) / 1000,
        firmicutes: Math.round(firmicutes * 1000) / 1000,
        proteobacteria: Math.round(proteobacteria * 1000) / 1000,
        actinobacteria: Math.round(actinobacteria * 1000) / 1000,
        diversity: Math.round(diversity * 1000) / 1000,
      });
    }
    
    const accuracy = 0.85 + Math.random() * 0.1;
    const stability = 75 + Math.random() * 20;
    
    return {
      data,
      metrics: {
        accuracy: Math.round(accuracy * 1000) / 1000,
        stability: Math.round(stability * 10) / 10,
      },
    };
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const { data, metrics } = generateMockTemporal(params.predictionSteps, params.interventionType);
    setTemporalData(data);
    setPredictionMetrics(metrics);
    
    setIsPredicting(false);
  };

  const getInterventionDescription = (intervention: string) => {
    switch (intervention) {
      case 'antibiotic':
        return 'Simulating the effects of antibiotic treatment on microbiome composition';
      case 'probiotic':
        return 'Modeling the impact of probiotic supplementation';
      case 'diet':
        return 'Predicting changes due to dietary modifications';
      default:
        return 'Natural microbiome evolution without intervention';
    }
  };

  return (
    <Box>
      <Typography
        variant="h3"
        component="h2"
        sx={{
          textAlign: 'center',
          marginBottom: '2rem',
          color: 'secondary.main',
          fontWeight: 600,
        }}
      >
        ⏰ Temporal Prediction
      </Typography>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card sx={{ marginBottom: '2rem', background: '#f8f9fa' }}>
          <CardContent sx={{ padding: '2rem' }}>
            <Grid container spacing={3} alignItems="flex-end">
              <Grid item xs={12} md={4}>
                <Typography gutterBottom>
                  Prediction Steps: {params.predictionSteps}
                </Typography>
                <Slider
                  value={params.predictionSteps}
                  onChange={(_, value) => setParams({ ...params, predictionSteps: value as number })}
                  min={1}
                  max={10}
                  marks
                  valueLabelDisplay="auto"
                  sx={{ color: 'primary.main' }}
                />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel>Intervention</InputLabel>
                  <Select
                    value={params.interventionType}
                    label="Intervention"
                    onChange={(e) => setParams({ ...params, interventionType: e.target.value as any })}
                  >
                    <MenuItem value="none">No Intervention</MenuItem>
                    <MenuItem value="antibiotic">Antibiotic Treatment</MenuItem>
                    <MenuItem value="probiotic">Probiotic Supplement</MenuItem>
                    <MenuItem value="diet">Dietary Change</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Button
                  variant="contained"
                  onClick={handlePredict}
                  disabled={isPredicting}
                  fullWidth
                  sx={{
                    padding: '1rem 2rem',
                    borderRadius: '25px',
                    fontWeight: 600,
                    background: 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 6px 20px rgba(52, 152, 219, 0.4)',
                    },
                  }}
                  startIcon={isPredicting ? <CircularProgress size={20} color="inherit" /> : null}
                >
                  {isPredicting ? 'Predicting...' : 'Predict Future States'}
                </Button>
              </Grid>
            </Grid>
            
            <Typography
              variant="body2"
              sx={{
                marginTop: '1rem',
                textAlign: 'center',
                color: 'text.secondary',
                fontStyle: 'italic',
              }}
            >
              {getInterventionDescription(params.interventionType)}
            </Typography>
          </CardContent>
        </Card>
      </motion.div>

      {temporalData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom textAlign="center">
                    Microbiome Evolution Over Time
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={temporalData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timePoint" label={{ value: 'Time Points', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'Relative Abundance', angle: -90, position: 'insideLeft' }} />
                      <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                      <Legend />
                      <Line type="monotone" dataKey="bacteroides" stroke="#3498db" strokeWidth={2} name="Bacteroides" />
                      <Line type="monotone" dataKey="firmicutes" stroke="#e74c3c" strokeWidth={2} name="Firmicutes" />
                      <Line type="monotone" dataKey="proteobacteria" stroke="#2ecc71" strokeWidth={2} name="Proteobacteria" />
                      <Line type="monotone" dataKey="actinobacteria" stroke="#f39c12" strokeWidth={2} name="Actinobacteria" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Card sx={{ textAlign: 'center' }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Prediction Accuracy
                      </Typography>
                      <Typography
                        variant="h3"
                        sx={{
                          fontWeight: 'bold',
                          color: predictionMetrics && predictionMetrics.accuracy > 0.9 ? '#27ae60' : '#f39c12',
                          marginBottom: '0.5rem',
                        }}
                      >
                        {predictionMetrics?.accuracy.toFixed(3)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Mean Absolute Error (lower is better)
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12}>
                  <Card sx={{ textAlign: 'center' }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Stability Score
                      </Typography>
                      <Typography
                        variant="h3"
                        sx={{
                          fontWeight: 'bold',
                          color: predictionMetrics && predictionMetrics.stability > 85 ? '#27ae60' : '#f39c12',
                          marginBottom: '0.5rem',
                        }}
                      >
                        {predictionMetrics?.stability.toFixed(1)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Prediction consistency across samples
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </motion.div>
      )}
    </Box>
  );
};

export default Temporal;