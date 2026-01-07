import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  CircularProgress,
} from '@mui/material';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ScatterChart, Scatter } from 'recharts';

interface EvaluationMetrics {
  mfd: number;
  alphaPValue: number;
  betaPValue: number;
  validityScore: number;
}

const Evaluation: React.FC = () => {
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);

  const generateMockEvaluation = (): EvaluationMetrics => {
    return {
      mfd: 0.15 + Math.random() * 0.1,
      alphaPValue: Math.random() * 0.5,
      betaPValue: Math.random() * 0.3,
      validityScore: 85 + Math.random() * 10,
    };
  };

  const handleRunEvaluation = async () => {
    setIsEvaluating(true);
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const results = generateMockEvaluation();
    setMetrics(results);
    
    setIsEvaluating(false);
  };

  const getMetricColor = (value: number, metric: string) => {
    switch (metric) {
      case 'mfd':
        return value < 0.2 ? '#27ae60' : value < 0.3 ? '#f39c12' : '#e74c3c';
      case 'pvalue':
        return value > 0.05 ? '#27ae60' : value > 0.01 ? '#f39c12' : '#e74c3c';
      case 'validity':
        return value > 90 ? '#27ae60' : value > 80 ? '#f39c12' : '#e74c3c';
      default:
        return '#3498db';
    }
  };

  const diversityData = [
    { name: 'Shannon', real: 2.5, generated: 2.3 },
    { name: 'Simpson', real: 0.8, generated: 0.75 },
    { name: 'Chao1', real: 45, generated: 42 },
    { name: 'ACE', real: 48, generated: 46 },
  ];

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
        📊 Evaluation Metrics
      </Typography>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card sx={{ marginBottom: '2rem', background: '#f8f9fa' }}>
          <CardContent sx={{ padding: '2rem', textAlign: 'center' }}>
            <Button
              variant="contained"
              onClick={handleRunEvaluation}
              disabled={isEvaluating}
              sx={{
                padding: '1rem 3rem',
                borderRadius: '25px',
                fontWeight: 600,
                fontSize: '1.1rem',
                background: 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: '0 6px 20px rgba(52, 152, 219, 0.4)',
                },
              }}
              startIcon={isEvaluating ? <CircularProgress size={20} color="inherit" /> : null}
            >
              {isEvaluating ? 'Running Evaluation...' : 'Run Evaluation'}
            </Button>
            <Typography variant="body2" sx={{ marginTop: '1rem', color: 'text.secondary' }}>
              Comparing generated samples against real microbiome data
            </Typography>
          </CardContent>
        </Card>
      </motion.div>

      {metrics && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Grid container spacing={3} sx={{ marginBottom: '2rem' }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ textAlign: 'center', height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Microbiome Fréchet Distance
                  </Typography>
                  <Typography
                    variant="h3"
                    sx={{
                      fontWeight: 'bold',
                      color: getMetricColor(metrics.mfd, 'mfd'),
                      marginBottom: '0.5rem',
                    }}
                  >
                    {metrics.mfd.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Lower values indicate better quality
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ textAlign: 'center', height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Alpha Diversity
                  </Typography>
                  <Typography
                    variant="h3"
                    sx={{
                      fontWeight: 'bold',
                      color: getMetricColor(metrics.alphaPValue, 'pvalue'),
                      marginBottom: '0.5rem',
                    }}
                  >
                    {metrics.alphaPValue.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    KS test p-value
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ textAlign: 'center', height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Beta Diversity
                  </Typography>
                  <Typography
                    variant="h3"
                    sx={{
                      fontWeight: 'bold',
                      color: getMetricColor(metrics.betaPValue, 'pvalue'),
                      marginBottom: '0.5rem',
                    }}
                  >
                    {metrics.betaPValue.toFixed(3)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Community dissimilarity p-value
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ textAlign: 'center', height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Compositional Validity
                  </Typography>
                  <Typography
                    variant="h3"
                    sx={{
                      fontWeight: 'bold',
                      color: getMetricColor(metrics.validityScore, 'validity'),
                      marginBottom: '0.5rem',
                    }}
                  >
                    {metrics.validityScore.toFixed(1)}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Valid compositions
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom textAlign="center">
                Diversity Comparison: Real vs Generated
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={diversityData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="real" fill="#3498db" name="Real Data" />
                  <Bar dataKey="generated" fill="#e74c3c" name="Generated Data" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </Box>
  );
};

export default Evaluation;