import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Chip,
} from '@mui/material';
import { motion } from 'framer-motion';

interface Method {
  id: string;
  name: string;
  enabled: boolean;
}

interface ComparisonResult {
  method: string;
  mfd: number;
  alphaDiversity: number;
  betaDiversity: number;
  generationSpeed: number;
  rank: number;
}

const Comparison: React.FC = () => {
  const [methods, setMethods] = useState<Method[]>([
    { id: 'diffusion', name: 'Diffusion Model (Ours)', enabled: true },
    { id: 'vae', name: 'Variational Autoencoder', enabled: true },
    { id: 'gan', name: 'Generative Adversarial Network', enabled: false },
    { id: 'baseline', name: 'Random Baseline', enabled: false },
  ]);

  const [isComparing, setIsComparing] = useState(false);
  const [comparisonResults, setComparisonResults] = useState<ComparisonResult[]>([]);

  const handleMethodToggle = (methodId: string) => {
    setMethods(prev =>
      prev.map(method =>
        method.id === methodId ? { ...method, enabled: !method.enabled } : method
      )
    );
  };

  const generateMockComparison = (enabledMethods: Method[]): ComparisonResult[] => {
    const methodPerformance = {
      diffusion: { mfd: 0.15, alpha: 0.85, beta: 0.78, speed: 120 },
      vae: { mfd: 0.22, alpha: 0.72, beta: 0.65, speed: 200 },
      gan: { mfd: 0.28, alpha: 0.68, beta: 0.62, speed: 150 },
      baseline: { mfd: 0.45, alpha: 0.45, beta: 0.40, speed: 1000 },
    };

    const results: ComparisonResult[] = enabledMethods.map((method) => {
      const perf = methodPerformance[method.id as keyof typeof methodPerformance];
      return {
        method: method.name,
        mfd: perf.mfd + (Math.random() - 0.5) * 0.02,
        alphaDiversity: perf.alpha + (Math.random() - 0.5) * 0.05,
        betaDiversity: perf.beta + (Math.random() - 0.5) * 0.05,
        generationSpeed: perf.speed + (Math.random() - 0.5) * 20,
        rank: 0,
      };
    });

    // Sort by MFD (lower is better) and assign ranks
    results.sort((a, b) => a.mfd - b.mfd);
    results.forEach((result, index) => {
      result.rank = index + 1;
    });

    return results;
  };

  const handleRunComparison = async () => {
    const enabledMethods = methods.filter(m => m.enabled);
    if (enabledMethods.length < 2) {
      alert('Please select at least 2 methods to compare');
      return;
    }

    setIsComparing(true);

    await new Promise(resolve => setTimeout(resolve, 2500));

    const results = generateMockComparison(enabledMethods);
    setComparisonResults(results);

    setIsComparing(false);
  };

  const getRankColor = (rank: number) => {
    switch (rank) {
      case 1: return '#FFD700';
      case 2: return '#C0C0C0';
      case 3: return '#CD7F32';
      default: return '#95a5a6';
    }
  };

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1: return '🥇';
      case 2: return '🥈';
      case 3: return '🥉';
      default: return `#${rank}`;
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
        ⚖️ Method Comparison
      </Typography>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card sx={{ marginBottom: '2rem', background: '#f8f9fa' }}>
          <CardContent sx={{ padding: '2rem' }}>
            <Typography variant="h6" gutterBottom>
              Select Methods to Compare:
            </Typography>
            <Grid container spacing={2} sx={{ marginBottom: '1.5rem' }}>
              {methods.map(method => (
                <Grid item xs={12} sm={6} md={3} key={method.id}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={method.enabled}
                        onChange={() => handleMethodToggle(method.id)}
                      />
                    }
                    label={method.name}
                  />
                </Grid>
              ))}
            </Grid>
            <Box sx={{ textAlign: 'center' }}>
              <Button
                variant="contained"
                onClick={handleRunComparison}
                disabled={isComparing || methods.filter(m => m.enabled).length < 2}
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
                startIcon={isComparing ? <CircularProgress size={20} color="inherit" /> : null}
              >
                {isComparing ? 'Running Comparison...' : 'Run Comparison'}
              </Button>
            </Box>
          </CardContent>
        </Card>
      </motion.div>

      {comparisonResults.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Comparison
              </Typography>
              <TableContainer component={Paper} sx={{ boxShadow: 'none' }}>
                <Table>
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#3498db' }}>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Rank</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Method</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">MFD ↓</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">Alpha Diversity</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">Beta Diversity</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">Generation Speed</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {comparisonResults.map((result, index) => (
                      <TableRow
                        key={index}
                        hover
                        sx={{
                          backgroundColor: result.rank === 1 ? 'rgba(255, 215, 0, 0.1)' : 'inherit',
                        }}
                      >
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span style={{ fontSize: '1.2rem' }}>{getRankIcon(result.rank)}</span>
                            <Chip
                              label={`#${result.rank}`}
                              size="small"
                              sx={{
                                backgroundColor: getRankColor(result.rank),
                                color: 'white',
                                fontWeight: 'bold',
                              }}
                            />
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography fontWeight={result.rank === 1 ? 'bold' : 'normal'}>
                            {result.method}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            color={result.mfd < 0.2 ? 'success.main' : result.mfd < 0.3 ? 'warning.main' : 'error.main'}
                            fontWeight="bold"
                          >
                            {result.mfd.toFixed(3)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">{result.alphaDiversity.toFixed(3)}</TableCell>
                        <TableCell align="right">{result.betaDiversity.toFixed(3)}</TableCell>
                        <TableCell align="right">{Math.round(result.generationSpeed)} samples/sec</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </Box>
  );
};

export default Comparison;