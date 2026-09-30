using UnityEngine;
using UnityEngine.Rendering;  // Required for AmbientMode
using System.Collections;

public class SceneLightingFix : MonoBehaviour
{
    [Header("Brightness Settings")]
    [Tooltip("How bright the ambient light is (works only in Flat mode)")]
    public float ambientIntensity = 2.0f;
    public Color ambientColor = Color.white;
    [Tooltip("Intensity of the main directional light")]
    public float directionalLightIntensity = 1.5f;

    void Start()
    {
        ApplyLighting();
        // Apply repeatedly to override any late changes (e.g., from Bootstrap)
        StartCoroutine(ApplyRepeatedly());
    }

    void ApplyLighting()
    {
        // Force Flat ambient mode so that ambientIntensity takes full effect
        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientIntensity = ambientIntensity;
        RenderSettings.ambientLight = ambientColor;

        // Also adjust the main directional light
        Light sun = FindObjectOfType<Light>();
        if (sun != null && sun.type == LightType.Directional)
        {
            sun.intensity = directionalLightIntensity;
        }

        Debug.Log($"[SceneLightingFix] Applied: ambient intensity = {ambientIntensity}, light intensity = {directionalLightIntensity}");
    }

    IEnumerator ApplyRepeatedly()
    {
        // Apply multiple times to catch any later overrides
        for (int i = 0; i < 5; i++)
        {
            yield return new WaitForSeconds(0.2f);
            ApplyLighting();
        }
    }
}