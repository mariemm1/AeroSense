import { Component } from '@angular/core';
import { ReactiveFormsModule, FormBuilder, Validators, FormGroup } from '@angular/forms';
import { RouterLink, Router } from '@angular/router';
import { NgIf } from '@angular/common';
import { AuthLayout } from '../auth-layout/auth-layout';
import { Auth } from '../services/authService/auth';

@Component({
  selector: 'app-signup',
  standalone: true,
  imports: [AuthLayout, ReactiveFormsModule, RouterLink, NgIf],
  templateUrl: './signup.html',
  styleUrls: ['./signup.css'],
})
export class Signup {
  submitted = false;
  loading = false;
  error = '';
  success = '';

  form: FormGroup;

  constructor(private fb: FormBuilder, private auth: Auth, private router: Router) {
    // role removed
    this.form = this.fb.group({
      full_name: ['', [Validators.required, Validators.minLength(2)]],
      what_he_does: ['', Validators.required],
      region: ['', Validators.required],
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]],
    });
  }

  submit() {
    this.error = '';
    this.success = '';
    this.submitted = true;
    if (this.form.invalid) return;

    this.loading = true;

    // backend accepts free-form role but it’s optional → not sent
    this.auth.signup(this.form.value as any).subscribe({
      next: () => {
        this.loading = false;
        // simple pop message then go to Signin layout
        alert('Account created. Please check your email to verify, then log in.');
        this.router.navigateByUrl('/signin');
      },
      error: (err) => {
        this.loading = false;
        this.error = err?.error?.error || 'Signup failed';
      },
    });
  }
}
