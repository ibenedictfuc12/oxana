package main

import (
    "fmt"
    "log"
    "net/http"
    "os/exec"
)

func main() {
    http.HandleFunc("/generate-image", func(w http.ResponseWriter, r *http.Request) {
        cmd := exec.Command("python", "examples/generate_image.py")
        output, err := cmd.CombinedOutput()
        if err != nil {
            fmt.Fprintf(w, "Error: %v\n", err)
            return
        }
        fmt.Fprintf(w, "Image generation result:\n%s", string(output))
    })

    fmt.Println("Go server started on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}