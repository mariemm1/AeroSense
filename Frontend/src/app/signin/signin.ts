import { Component } from '@angular/core';
import { ReactiveFormsModule, FormBuilder, Validators, FormGroup } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { NgIf } from '@angular/common';
import { AuthLayout } from '../auth-layout/auth-layout';
import { Auth } from '../services/authService/auth';

@Component({
  selector: 'app-signin',
  standalone: true,
  imports: [AuthLayout, ReactiveFormsModule, RouterLink, NgIf],
  templateUrl: './signin.html',
  styleUrls: ['./signin.css'],
})
export class Signin {
  submitted = false;
  loading = false;
  needsVerification = false;
  error = '';

  form: FormGroup;

  constructor(private fb: FormBuilder, private auth: Auth, private router: Router) {
    // ✅ create form in constructor (no “used before initialization”)
    this.form = this.fb.group({
      username: ['', Validators.required],
      password: ['', Validators.required],
    });
  }

  submit() {
    this.submitted = true;
    this.error = '';
    if (this.form.invalid) return;
    this.loading = true;
    this.needsVerification = false;

    this.auth.signin(this.form.value as any).subscribe({
      next: (res) => {
        this.loading = false;
        if (res?.token) {
          this.auth.saveToken(res.token);
          this.router.navigateByUrl('/dashboard');
        } else if (res?.needs_verification) {
          this.needsVerification = true;
          this.error = 'Please verify your email first.';
        } else {
          this.error = res?.error || 'Sign in failed';
        }
      },
      error: (err) => {
        this.loading = false;
        const e = err?.error;
        if (e?.needs_verification) {
          this.needsVerification = true;
          this.error = 'Please verify your email first.';
        } else {
          this.error = e?.error || 'Sign in failed';
        }
      },
    });
  }

  resend() {
    const u = this.form.controls['username'].value || '';
    this.auth.resendVerification(u).subscribe({
      next: () => (this.error = 'Verification email sent. Please check your inbox.'),
      error: (err) => (this.error = err?.error?.error || 'Failed to resend'),
    });
  }
}
