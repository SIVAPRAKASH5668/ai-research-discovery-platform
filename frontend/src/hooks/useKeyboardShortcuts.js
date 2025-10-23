import { useEffect } from 'react'

export const useKeyboardShortcuts = (shortcuts) => {
  useEffect(() => {
    // DISABLED: Keyboard shortcuts are turned off
    // No event listener attached, so typing works normally
    return () => {
      // Cleanup (nothing to clean up)
    }
  }, [shortcuts])
}
