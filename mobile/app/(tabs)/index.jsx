import React from 'react';
import { View, Text, ScrollView, Pressable, StyleSheet, SafeAreaView, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient'; // Ensure this is installed or use fallback View

// Fallback for LinearGradient if not installed, though it's standard in Expo
const Gradient = ({ children, style, colors }) => {
  try {
    const { LinearGradient } = require('expo-linear-gradient');
    return <LinearGradient colors={colors} style={style} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}>{children}</LinearGradient>;
  } catch (e) {
    return <View style={[style, { backgroundColor: colors[0] }]}>{children}</View>;
  }
};

export default function Home() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

        {/* Header Section */}
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>Welcome back,</Text>
            <Text style={styles.appName}>CoughAI</Text>
          </View>
          <View style={styles.avatarPlaceholder}>
            <Feather name="user" size={24} color="#2563EB" />
          </View>
        </View>

        {/* Hero Card */}
        <Pressable onPress={() => router.push('/(tabs)/cough')} style={({ pressed }) => [styles.heroCard, pressed && styles.pressed]}>
          <Gradient colors={['#2563EB', '#1D4ED8']} style={styles.heroGradient}>
            <View style={styles.heroContent}>
              <View style={styles.heroIconContainer}>
                <Feather name="mic" size={32} color="#FFFFFF" />
              </View>
              <View style={styles.heroTextContainer}>
                <Text style={styles.heroTitle}>Start Analysis</Text>
                <Text style={styles.heroSubtitle}>Record cough samples for AI health assessment</Text>
              </View>
              <Feather name="arrow-right" size={24} color="#FFFFFF" style={styles.heroArrow} />
            </View>
          </Gradient>
        </Pressable>

        {/* Stats Row */}
        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <View style={[styles.statIcon, { backgroundColor: '#EFF6FF' }]}>
              <Feather name="activity" size={20} color="#2563EB" />
            </View>
            <Text style={styles.statValue}>Healthy</Text>
            <Text style={styles.statLabel}>Last Result</Text>
          </View>
          <View style={styles.statCard}>
            <View style={[styles.statIcon, { backgroundColor: '#F0FDF4' }]}>
              <Feather name="check-circle" size={20} color="#16A34A" />
            </View>
            <Text style={styles.statValue}>Online</Text>
            <Text style={styles.statLabel}>System Status</Text>
          </View>
        </View>

        {/* Recent Activity Section */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Recent Activity</Text>
          <Pressable>
            <Text style={styles.seeAll}>See All</Text>
          </Pressable>
        </View>

        <View style={styles.emptyStateContainer}>
          <View style={styles.emptyStateIcon}>
            <Feather name="clipboard" size={40} color="#94A3B8" />
          </View>
          <Text style={styles.emptyStateTitle}>No recent analysis</Text>
          <Text style={styles.emptyStateText}>Your health evaluations will appear here.</Text>
        </View>

        {/* Information Section */}
        <View style={styles.infoCard}>
          <View style={styles.infoHeader}>
            <Feather name="info" size={20} color="#1E40AF" />
            <Text style={styles.infoTitle}>How it works</Text>
          </View>
          <Text style={styles.infoText}>
            Our advanced AI analyzes audio patterns in your cough to detect potential respiratory conditions.
            Recordings are processed securely ensuring your privacy.
          </Text>
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8FAFC', // Slate 50
    paddingTop: Platform.OS === 'android' ? 40 : 0,
  },
  scrollContent: {
    padding: 24,
    paddingBottom: 40,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 32,
  },
  greeting: {
    fontSize: 16,
    color: '#64748B', // Slate 500
    marginBottom: 4,
  },
  appName: {
    fontSize: 28,
    fontWeight: '700',
    color: '#0F172A', // Slate 900
    letterSpacing: -0.5,
  },
  avatarPlaceholder: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#EFF6FF', // Blue 50
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#DBEAFE',
  },
  heroCard: {
    borderRadius: 24,
    marginBottom: 32,
    shadowColor: '#2563EB',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 16,
    elevation: 8,
  },
  heroGradient: {
    borderRadius: 24,
    padding: 24,
  },
  heroContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  heroIconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  heroTextContainer: {
    flex: 1,
  },
  heroTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  heroSubtitle: {
    fontSize: 13,
    color: '#DBEAFE', // Blue 100
    lineHeight: 18,
  },
  pressed: {
    opacity: 0.9,
    transform: [{ scale: 0.98 }],
  },
  statsRow: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 32,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#F1F5F9', // Slate 100
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.03,
    shadowRadius: 8,
    elevation: 2,
  },
  statIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  statValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#0F172A',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 13,
    color: '#64748B',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#0F172A',
  },
  seeAll: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2563EB',
  },
  emptyStateContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    padding: 32,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#F1F5F9',
    marginBottom: 32,
    borderStyle: 'dashed',
  },
  emptyStateIcon: {
    marginBottom: 16,
    opacity: 0.5,
  },
  emptyStateTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#0F172A',
    marginBottom: 8,
  },
  emptyStateText: {
    fontSize: 14,
    color: '#64748B',
    textAlign: 'center',
  },
  infoCard: {
    backgroundColor: '#EFF6FF', // Blue 50
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: '#DBEAFE',
  },
  infoHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1E40AF', // Blue 800
    marginLeft: 8,
  },
  infoText: {
    fontSize: 14,
    color: '#1E40AF',
    lineHeight: 22,
    opacity: 0.8,
  },
});