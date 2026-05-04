import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, Pressable, StyleSheet, Alert, SafeAreaView, Platform, ActivityIndicator } from 'react-native';
import { useAuth, useUser } from '@clerk/clerk-expo';
import Constants from 'expo-constants';
import { Feather } from '@expo/vector-icons';
import axios from 'axios';
import { useFocusEffect } from 'expo-router';

export default function Profile() {
  const { signOut, getToken } = useAuth();
  const { user } = useUser();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const API_URL = Constants.expoConfig?.extra?.API_URL || 'http://localhost:4000';

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const token = await getToken();
      const response = await axios.get(`${API_URL}/api/history`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHistory(response.data.history || []);
    } catch (error) {
      console.error('Failed to fetch history:', error);
      // Don't alert on load, just log
    } finally {
      setLoading(false);
    }
  };

  useFocusEffect(
    React.useCallback(() => {
      fetchHistory();
    }, [])
  );

  const handleSignOut = () => {
    signOut();
  };

  const formatDate = (dateString) => {
    const d = new Date(dateString);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const HistoryItem = ({ item }) => {
    const data = item.results_json;
    const isHealthy = data.probabilities?.[0]?.disease === 'Healthy' && data.probabilities[0].probability > 80;

    return (
      <View style={styles.historyCard}>
        <View style={styles.historyHeader}>
          <View style={[styles.statusDot, { backgroundColor: isHealthy ? '#16A34A' : '#EAB308' }]} />
          <Text style={styles.historyDate}>{formatDate(item.created_at)}</Text>
        </View>
        <Text style={styles.historyResult}>
          {data.probabilities?.[0]?.disease || 'Analysis'} ({data.probabilities?.[0]?.probability}%)
        </Text>
      </View>
    );
  };

  const MenuItem = ({ icon, label, onPress, isLast }) => (
    <Pressable style={[styles.menuItem, isLast && styles.menuItemLast]} onPress={onPress}>
      <View style={styles.menuItemLeft}>
        <Feather name={icon} size={20} color="#64748B" />
        <Text style={styles.menuItemText}>{label}</Text>
      </View>
      <Feather name="chevron-right" size={20} color="#94A3B8" />
    </Pressable>
  );

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Profile</Text>
        </View>

        <View style={styles.profileCard}>
          <View style={styles.avatarContainer}>
            <View style={styles.avatar}>
              {user?.imageUrl ? (
                // In a real app use <Image> but for now use icon or assume simple
                <Feather name="user" size={32} color="#2563EB" />
              ) : (
                <Feather name="user" size={32} color="#2563EB" />
              )}
            </View>
            <View style={styles.badge}>
              <Feather name="check" size={12} color="#FFFFFF" />
            </View>
          </View>
          <Text style={styles.profileName}>{user?.fullName || 'User'}</Text>
          <Text style={styles.profileEmail}>{user?.primaryEmailAddress?.emailAddress || 'No email'}</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>History</Text>
          {loading ? (
            <ActivityIndicator color="#2563EB" style={{ marginTop: 20 }} />
          ) : history.length > 0 ? (
            <View style={styles.historyList}>
              {history.map(item => <HistoryItem key={item.id} item={item} />)}
            </View>
          ) : (
            <View style={styles.emptyState}>
              <Feather name="clock" size={40} color="#CBD5E1" />
              <Text style={styles.emptyText}>No analysis history yet</Text>
            </View>
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account</Text>
          <View style={styles.menuCard}>
            <MenuItem icon="user" label="Personal Information" />
            <MenuItem icon="lock" label="Privacy Settings" />
            <MenuItem icon="bell" label="Notifications" isLast />
          </View>
        </View>

        <Pressable onPress={handleSignOut} style={styles.signOutButton}>
          <Feather name="log-out" size={20} color="#EF4444" style={styles.signOutIcon} />
          <Text style={styles.signOutText}>Sign Out</Text>
        </Pressable>

        <Text style={styles.versionText}>Version 1.0.0 (Pro)</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F8FAFC', paddingTop: Platform.OS === 'android' ? 40 : 0 },
  scrollContent: { padding: 24 },
  header: { marginBottom: 24, paddingVertical: 12 },
  headerTitle: { fontSize: 24, fontWeight: '700', color: '#0F172A' },

  profileCard: { alignItems: 'center', marginBottom: 32 },
  avatarContainer: { marginBottom: 16, position: 'relative' },
  avatar: { width: 80, height: 80, borderRadius: 40, backgroundColor: '#EFF6FF', alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: '#DBEAFE' },
  badge: { position: 'absolute', bottom: 0, right: 0, backgroundColor: '#16A34A', borderRadius: 10, width: 24, height: 24, alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: '#F8FAFC' },
  profileName: { fontSize: 20, fontWeight: '700', color: '#0F172A', marginBottom: 4 },
  profileEmail: { fontSize: 14, color: '#64748B' },

  section: { marginBottom: 24 },
  sectionTitle: { fontSize: 13, fontWeight: '600', color: '#64748B', marginBottom: 12, marginLeft: 4, textTransform: 'uppercase', letterSpacing: 0.5 },

  menuCard: { backgroundColor: '#FFFFFF', borderRadius: 16, paddingLeft: 16, paddingRight: 8, borderWidth: 1, borderColor: '#F1F5F9' },
  menuItem: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 16, borderBottomWidth: 1, borderBottomColor: '#F1F5F9' },
  menuItemLast: { borderBottomWidth: 0 },
  menuItemLeft: { flexDirection: 'row', alignItems: 'center' },
  menuItemText: { fontSize: 16, color: '#0F172A', marginLeft: 16, fontWeight: '500' },

  signOutButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', backgroundColor: '#FEF2F2', borderRadius: 16, padding: 16, borderWidth: 1, borderColor: '#FECACA', marginBottom: 24 },
  signOutIcon: { marginRight: 8 },
  signOutText: { color: '#DC2626', fontWeight: '600', fontSize: 16 },

  versionText: { textAlign: 'center', color: '#94A3B8', fontSize: 12 },

  historyList: { gap: 12 },
  historyCard: { backgroundColor: 'white', padding: 16, borderRadius: 16, borderWidth: 1, borderColor: '#F1F5F9' },
  historyHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  statusDot: { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  historyDate: { color: '#64748B', fontSize: 12 },
  historyResult: { color: '#0F172A', fontSize: 16, fontWeight: '600' },

  emptyState: { alignItems: 'center', padding: 32, backgroundColor: 'white', borderRadius: 16, borderWidth: 1, borderColor: '#F1F5F9' },
  emptyText: { color: '#94A3B8', marginTop: 12, fontSize: 14 }
});