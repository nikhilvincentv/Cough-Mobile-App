import { Tabs } from 'expo-router';
import { StyleSheet, View } from 'react-native';
import { Feather } from '@expo/vector-icons';

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: '#FFFFFF',
    borderTopWidth: 1,
    borderTopColor: '#F3F4F6',
    height: 90,
    paddingTop: 10,
    paddingHorizontal: 10,
    elevation: 0,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
  },
  tabLabel: {
    fontSize: 12,
    fontWeight: '500',
    marginTop: 4,
    marginBottom: 4,
  },
});

export default function TabsLayout() {
  return (
    <Tabs 
      screenOptions={{ 
        headerShown: false, 
        tabBarActiveTintColor: '#2563EB', // blue-600
        tabBarInactiveTintColor: '#94A3B8', // slate-400
        tabBarStyle: styles.tabBar,
        tabBarLabelStyle: styles.tabLabel,
        tabBarShowLabel: true,
      }}
    >
      <Tabs.Screen 
        name="index" 
        options={{ 
          title: 'Home',
          tabBarIcon: ({ color, size }) => (
            <Feather name="home" size={24} color={color} />
          )
        }} 
      />
      <Tabs.Screen 
        name="cough/index" 
        options={{ 
          title: 'Analysis',
          tabBarIcon: ({ color, size }) => (
            <View style={{
              backgroundColor: color === '#2563EB' ? '#EFF6FF' : 'transparent',
              padding: 8,
              borderRadius: 12,
            }}>
              <Feather name="activity" size={24} color={color} />
            </View>
          )
        }} 
      />
      <Tabs.Screen 
        name="profile" 
        options={{ 
          title: 'Profile',
          tabBarIcon: ({ color, size }) => (
            <Feather name="user" size={24} color={color} />
          )
        }} 
      />
    </Tabs>
  );
}