import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut, onAuthStateChanged } from "firebase/auth";

// Shared portfolio Firebase project ("cabbage-guard"). Public web config —
// safe to ship client-side; access control lives server-side (ID-token
// verification) and in Firestore/Storage rules, not here.
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

export function getFirebaseApp() {
  return getApps().length ? getApp() : initializeApp(firebaseConfig);
}

export function getFirebaseAuth() {
  return getAuth(getFirebaseApp());
}

export function signIn() {
  return signInWithPopup(getFirebaseAuth(), new GoogleAuthProvider());
}

export function signOutUser() {
  return signOut(getFirebaseAuth());
}

export function watchAuth(cb) {
  return onAuthStateChanged(getFirebaseAuth(), cb);
}
