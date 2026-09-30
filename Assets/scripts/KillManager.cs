using UnityEngine;

public class KillManager : MonoBehaviour
{
    public static KillManager Instance { get; private set; }

    public int killCount { get; private set; } = 0;

    // Event for UI updates
    public System.Action<int> OnKillCountChanged;

    private void Awake()
    {
        // Singleton
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    public void AddKill()
    {
        killCount++;
        OnKillCountChanged?.Invoke(killCount);
        Debug.Log($"Enemy killed. Total kills: {killCount}");
    }

    public void ResetKills()
    {
        killCount = 0;
        OnKillCountChanged?.Invoke(killCount);
    }
}