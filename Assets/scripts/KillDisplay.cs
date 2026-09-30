using UnityEngine;
using TMPro;
using System.Collections;

public class KillDisplay : MonoBehaviour
{
    public TMP_Text killText;   // assign in Inspector
    private Coroutine popRoutine;

    private void Start()
    {
        if (KillManager.Instance == null)
        {
            Debug.LogWarning("KillManager not found.");
            return;
        }

        // Set initial text
        UpdateText(KillManager.Instance.killCount);
        // Subscribe to changes
        KillManager.Instance.OnKillCountChanged += UpdateText;
    }

    private void UpdateText(int count)
    {
        killText.text = $"KILLS: <color=#8B0000>{count}</color>";

        if (popRoutine != null)
            StopCoroutine(popRoutine);

        popRoutine = StartCoroutine(PopAnimation());
    }

    private IEnumerator PopAnimation()
    {
        float duration = 0.30f;

        Vector3 startScale = Vector3.one;
        Vector3 peakScale = Vector3.one * 1.5f;

        float t = 0f;

        // Scale up
        while (t < duration * 0.5f)
        {
            t += Time.deltaTime;
            float p = t / (duration * 0.5f);

            transform.localScale =
                Vector3.Lerp(startScale, peakScale, p);

            yield return null;
        }

        t = 0f;

        // Scale down
        while (t < duration * 0.5f)
        {
            t += Time.deltaTime;
            float p = t / (duration * 0.5f);

            transform.localScale =
                Vector3.Lerp(peakScale, startScale, p);

            yield return null;
        }

        // Guarantee final scale
        transform.localScale = Vector3.one;
    }

    private void OnDestroy()
    {
        if (KillManager.Instance != null)
            KillManager.Instance.OnKillCountChanged -= UpdateText;
    }
}