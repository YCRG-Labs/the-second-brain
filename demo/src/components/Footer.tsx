import React from 'react';
import { Box, Typography, Link, Container, IconButton } from '@mui/material';
import { motion } from 'framer-motion';

const Footer: React.FC = () => {
  return (
    <Box
      component={motion.footer}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8, delay: 0.5 }}
      sx={{
        background: 'rgba(15, 15, 35, 0.9)',
        backdropFilter: 'blur(20px)',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        padding: '3rem 0 2rem',
        textAlign: 'center',
        marginTop: 'auto',
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '1px',
          background: 'linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent)',
        },
      }}
    >
      <Container maxWidth="lg">
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.7 }}
        >
          <Typography
            variant="h6"
            sx={{
              color: 'white',
              marginBottom: '1rem',
              fontWeight: 600,
            }}
          >
            🧠 The Second Brain
          </Typography>
          
          <Typography
            variant="body1"
            sx={{
              color: 'rgba(255, 255, 255, 0.7)',
              marginBottom: '2rem',
              maxWidth: '600px',
              margin: '0 auto 2rem',
              lineHeight: 1.6,
            }}
          >
            Advancing microbiome research through cutting-edge AI and machine learning techniques.
            This interactive demo showcases our novel approach to generating realistic microbiome compositions.
          </Typography>
        </motion.div>

        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.9 }}
        >
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'center',
              gap: '2rem',
              flexWrap: 'wrap',
              marginBottom: '2rem',
            }}
          >
            {[
              { label: 'GitHub Repository', href: 'https://github.com/your-repo', icon: '📂' },
              { label: 'Research Paper', href: '#', icon: '📄' },
              { label: 'Documentation', href: '#', icon: '📚' },
              { label: 'Contact Us', href: '#', icon: '✉️' },
            ].map((link, index) => (
              <motion.div
                key={link.label}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    color: 'rgba(255, 255, 255, 0.8)',
                    textDecoration: 'none',
                    padding: '0.75rem 1.5rem',
                    borderRadius: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    '&:hover': {
                      color: 'white',
                      background: 'rgba(255, 255, 255, 0.1)',
                      borderColor: 'rgba(255, 255, 255, 0.2)',
                      transform: 'translateY(-2px)',
                    },
                  }}
                >
                  <span>{link.icon}</span>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {link.label}
                  </Typography>
                </Link>
              </motion.div>
            ))}
          </Box>
        </motion.div>

        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 1.1 }}
        >
          <Box
            sx={{
              borderTop: '1px solid rgba(255, 255, 255, 0.1)',
              paddingTop: '2rem',
              display: 'flex',
              flexDirection: { xs: 'column', md: 'row' },
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '1rem',
            }}
          >
            <Typography
              variant="body2"
              sx={{
                color: 'rgba(255, 255, 255, 0.5)',
                fontSize: '0.9rem',
              }}
            >
              &copy; 2024 Yee Collins Research Group. All rights reserved.
            </Typography>
            
            <Typography
              variant="body2"
              sx={{
                color: 'rgba(255, 255, 255, 0.5)',
                fontSize: '0.9rem',
                fontStyle: 'italic',
              }}
            >
              Built with React, TypeScript & Material-UI
            </Typography>
          </Box>
        </motion.div>

        {/* Decorative elements */}
        <Box
          sx={{
            position: 'absolute',
            bottom: '20%',
            left: '5%',
            width: '60px',
            height: '60px',
            background: 'radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%)',
            borderRadius: '50%',
            filter: 'blur(15px)',
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            top: '30%',
            right: '10%',
            width: '40px',
            height: '40px',
            background: 'radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%)',
            borderRadius: '50%',
            filter: 'blur(10px)',
          }}
        />
      </Container>
    </Box>
  );
};

export default Footer;